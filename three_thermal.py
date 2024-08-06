# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

import torch
from sympy import Symbol, Eq, Abs, tanh, Or, And
import itertools
import numpy as np

import modulus.sym
from modulus.sym.hydra.config import ModulusConfig
from modulus.sym.hydra import to_absolute_path, instantiate_arch
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_3d import Box, Channel, Plane
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym.domain.inferencer import PointVTKInferencer
from modulus.sym.utils.io import (
    VTKUniformGrid,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec, GradNormal
from modulus.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from modulus.sym.eq.pdes.advection_diffusion import AdvectionDiffusion

from three_geometry import *
import scipy.interpolate 
import matplotlib.pyplot as plt

from modulus.sym.utils.io.plotter import ValidatorPlotter

# define custom class
class CustomValidatorPlotter(ValidatorPlotter):

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables
        x,y = invar["x"][:,0], invar["y"][:,0]
        extent = (x.min(), x.max(), y.min(), y.max())

        # get and interpolate output variable
        u_pred = pred_outvar["u"][:,0]
        u_pred = self.interpolate_output(x, y,
                                            u_pred,
                                            extent,
        )

        # make plot
        f = plt.figure(figsize=(14,4), dpi=100)
        plt.suptitle("Solution of the heat equation")
        plt.subplot(1,3,1)
        
        
        plt.subplot(1,3,2)
        plt.title("PINN solution (u)")
        plt.imshow(u_pred.T, origin="lower", extent=extent, vmin=-0.2, vmax=1)
        plt.xlabel("x"); plt.ylabel("y")
        plt.colorbar()
        plt.tight_layout()

        return [(f, "custom_plot"),]

    @staticmethod
    def interpolate_output(x, y, us, extent):
        "Interpolates irregular points onto a mesh"

        # define mesh to interpolate onto
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], 100),
            np.linspace(extent[2], extent[3], 100),
            indexing="ij",
        )

        # linearly interpolate points onto mesh
        us = [scipy.interpolate.griddata(
            (x, y), u, tuple(xyi)
            )
            for u in us]

        return us

@modulus.sym.main(config_path="conf", config_name="conf_thermal")
def run(cfg: ModulusConfig) -> None:
    # make thermal equations
    ad = AdvectionDiffusion(T="theta_f", rho=1.0, D=0.02, dim=3, time=False)
    dif = Diffusion(T="theta_s", D=0.0625, dim=3, time=False)
    dif_inteface = DiffusionInterface("theta_f", "theta_s", 1.0, 5.0, dim=3, time=False)
    f_grad = GradNormal("theta_f", dim=3, time=False)
    s_grad = GradNormal("theta_s", dim=3, time=False)

    # make network arch
    
    input_keys = [Key("x"), Key("y"), Key("z")]
    flow_net = FullyConnectedArch(
        input_keys=input_keys,
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
    )
    thermal_f_net = FullyConnectedArch(
        input_keys=input_keys, output_keys=[Key("theta_f")]
    )
    thermal_s_net = FullyConnectedArch(
        input_keys=input_keys, output_keys=[Key("theta_s")]
    )

    # make list of nodes to unroll graph on
    thermal_nodes = (
        ad.make_nodes()
        + dif.make_nodes()
        + dif_inteface.make_nodes()
        + f_grad.make_nodes()
        + s_grad.make_nodes()
        + [flow_net.make_node(name="flow_network", optimize=False)]
        + [thermal_f_net.make_node(name="thermal_f_network")]
        + [thermal_s_net.make_node(name="thermal_s_network")]
    )

    geo = ThreeFin()

    # params for simulation
    # heat params
    inlet_t = 293.15 / 273.15 - 1.0
    grad_t = 360 / 273.15

    # make flow domain
    thermal_domain = Domain()

    # inlet
    constraint_inlet = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.inlet,
        outvar={"theta_f": inlet_t},
        batch_size=cfg.batch_size.Inlet,
        criteria=Eq(x, channel_origin[0]),
        lambda_weighting={"theta_f": 1.0},  # weight zero on edges
    )
    thermal_domain.add_constraint(constraint_inlet, "inlet")

    # outlet
    constraint_outlet = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.outlet,
        outvar={"normal_gradient_theta_f": 0},
        batch_size=cfg.batch_size.Outlet,
        criteria=Eq(x, channel_origin[0] + channel_dim[0]),
        lambda_weighting={"normal_gradient_theta_f": 1.0},  # weight zero on edges
    )
    thermal_domain.add_constraint(constraint_outlet, "outlet")

    # channel walls insulating
    def wall_criteria(invar, params):
        sdf = geo.three_fin.sdf(invar, params)
        return np.less(sdf["sdf"], -1e-5)

    channel_walls = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.channel,
        outvar={"normal_gradient_theta_f": 0},
        batch_size=cfg.batch_size.ChannelWalls,
        criteria=wall_criteria,
        lambda_weighting={"normal_gradient_theta_f": 1.0},
    )
    thermal_domain.add_constraint(channel_walls, "channel_walls")

    # fluid solid interface
    def interface_criteria(invar, params):
        sdf = geo.channel.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    fluid_solid_interface = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.three_fin,
        outvar={
            "diffusion_interface_dirichlet_theta_f_theta_s": 0,
            "diffusion_interface_neumann_theta_f_theta_s": 0,
        },
        batch_size=cfg.batch_size.SolidInterface,
        criteria=interface_criteria,
    )
    thermal_domain.add_constraint(fluid_solid_interface, "fluid_solid_interface")

    # heat source
    sharpen_tanh = 60.0
    source_func_xl = (tanh(sharpen_tanh * (x - source_origin[0])) + 1.0) / 2.0
    source_func_xh = (
        tanh(sharpen_tanh * ((source_origin[0] + source_dim[0]) - x)) + 1.0
    ) / 2.0
    source_func_zl = (tanh(sharpen_tanh * (z - source_origin[2])) + 1.0) / 2.0
    source_func_zh = (
        tanh(sharpen_tanh * ((source_origin[2] + source_dim[2]) - z)) + 1.0
    ) / 2.0
    gradient_normal = (
        grad_t * source_func_xl * source_func_xh * source_func_zl * source_func_zh
    )
    heat_source = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.three_fin,
        outvar={"normal_gradient_theta_s": gradient_normal},
        batch_size=cfg.batch_size.HeatSource,
        criteria=Eq(y, source_origin[1]),
    )
    thermal_domain.add_constraint(heat_source, "heat_source")

    # flow interior low res away from three fin
    lr_flow_interior = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=geo.geo,
        outvar={"advection_diffusion_theta_f": 0},
        batch_size=cfg.batch_size.InteriorLR,
        criteria=Or(x < -1.1, x > 0.5),
    )
    thermal_domain.add_constraint(lr_flow_interior, "lr_flow_interior")

    # flow interiror high res near three fin
    hr_flow_interior = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=geo.geo,
        outvar={"advection_diffusion_theta_f": 0},
        batch_size=cfg.batch_size.InteriorHR,
        criteria=And(x > -1.1, x < 0.5),
    )
    thermal_domain.add_constraint(hr_flow_interior, "hr_flow_interior")

    # solid interior
    solid_interior = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=geo.three_fin,
        outvar={"diffusion_theta_s": 0},
        batch_size=cfg.batch_size.SolidInterior,
        lambda_weighting={"diffusion_theta_s": 100.0},
    )
    thermal_domain.add_constraint(solid_interior, "solid_interior")

    # add solid inferencer data
    vtk_obj = VTKUniformGrid(
        bounds=[
            geo.hr_bounds[x],
            geo.hr_bounds[y],
            geo.hr_bounds[z],
        ],
        npoints=[128, 128, 512],
        export_map={"theta_s": ["theta_s"], "theta_f": ["theta_f"] },
    )

    def mask_fn(x, y, z):
        sdf = geo.channel.sdf({"x": x, "y": y, "z": z}, {})
        return sdf["sdf"] < 0

    grid_inferencer = PointVTKInferencer(
        vtk_obj=vtk_obj,
        nodes=thermal_nodes,
        input_vtk_map={"x": "x", "y": "y", "z": "z"},
        output_names=["theta_f", "theta_s"],
        mask_fn=mask_fn,
        mask_value=np.nan,
        requires_grad=False,
        batch_size=100000,
    )
    thermal_domain.add_inferencer(grid_inferencer, "grid_inferencer_solid")

    '''
    plot_validator = PointwiseValidator(
        plotter=CustomValidatorPlotter(),
        nodes=thermal_nodes,
        invar={"x": "x", "y": "y", "z": "z"},
        true_outvar={},
    )
    thermal_domain.add_validator(plot_validator, "plot_validator")
    '''


    # flow validation data
    file_path = "../openfoam/"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "Points:0": "x",
            "Points:1": "y",
            "Points:2": "z",
            "U:0": "u",
            "U:1": "v",
            "U:2": "w",
            "p_rgh": "p",
            "T": "theta_f",
        }
        if cfg.custom.turbulent:
            openfoam_var = csv_to_dict(
                to_absolute_path("openfoam/threeFin_extend_zeroEq_re500_fluid.csv"),
                mapping,
            )
        else:
            openfoam_var = csv_to_dict(
                to_absolute_path("openfoam/threeFin_extend_fluid0.csv"), mapping
            )
        openfoam_var["theta_f"] = (
            openfoam_var["theta_f"] / 273.15 - 1.0
        )  # normalize heat
        openfoam_var["x"] = openfoam_var["x"] + channel_origin[0]
        openfoam_var["y"] = openfoam_var["y"] + channel_origin[1]
        openfoam_var["z"] = openfoam_var["z"] + channel_origin[2]
        openfoam_var.update({"fin_height_m": np.full_like(openfoam_var["x"], 0.4)})
        openfoam_var.update({"fin_height_s": np.full_like(openfoam_var["x"], 0.4)})
        openfoam_var.update({"fin_thickness_m": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"fin_thickness_s": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"fin_length_m": np.full_like(openfoam_var["x"], 1.0)})
        openfoam_var.update({"fin_length_s": np.full_like(openfoam_var["x"], 1.0)})
        openfoam_invar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key
            in [
                "x",
                "y",
                "z",
                "fin_height_m",
                "fin_height_s",
                "fin_thickness_m",
                "fin_thickness_s",
                "fin_length_m",
                "fin_length_s",
            ]
        }
        openfoam_flow_outvar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["u", "v", "w", "p"]
        }
        openfoam_thermal_outvar_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key in ["u", "v", "w", "p", "theta_f"]
        }
        openfoam_flow_validator = PointwiseValidator(
            nodes=thermal_nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_thermal_outvar_numpy,
        )
        thermal_domain.add_validator(
            openfoam_flow_validator,
            "thermal_flow_data",
        )

        # solid data
        mapping = {"Points:0": "x", "Points:1": "y", "Points:2": "z", "T": "theta_s"}
        if cfg.custom.turbulent:
            openfoam_var = csv_to_dict(
                to_absolute_path("openfoam/threeFin_extend_zeroEq_re500_solid.csv"),
                mapping,
            )
        else:
            openfoam_var = csv_to_dict(
                to_absolute_path("openfoam/threeFin_extend_solid0.csv"), mapping
            )
        openfoam_var["theta_s"] = (
            openfoam_var["theta_s"] / 273.15 - 1.0
        )  # normalize heat
        openfoam_var["x"] = openfoam_var["x"] + channel_origin[0]
        openfoam_var["y"] = openfoam_var["y"] + channel_origin[1]
        openfoam_var["z"] = openfoam_var["z"] + channel_origin[2]
        openfoam_var.update({"fin_height_m": np.full_like(openfoam_var["x"], 0.4)})
        openfoam_var.update({"fin_height_s": np.full_like(openfoam_var["x"], 0.4)})
        openfoam_var.update({"fin_thickness_m": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"fin_thickness_s": np.full_like(openfoam_var["x"], 0.1)})
        openfoam_var.update({"fin_length_m": np.full_like(openfoam_var["x"], 1.0)})
        openfoam_var.update({"fin_length_s": np.full_like(openfoam_var["x"], 1.0)})
        openfoam_invar_solid_numpy = {
            key: value
            for key, value in openfoam_var.items()
            if key
            in [
                "x",
                "y",
                "z",
                "fin_height_m",
                "fin_height_s",
                "fin_thickness_m",
                "fin_thickness_s",
                "fin_length_m",
                "fin_length_s",
            ]
        }
        openfoam_outvar_solid_numpy = {
            key: value for key, value in openfoam_var.items() if key in ["theta_s"]
        }
        openfoam_solid_validator = PointwiseValidator(
            nodes=thermal_nodes,
            invar=openfoam_invar_solid_numpy,
            true_outvar=openfoam_thermal_outvar_numpy,
        )
        thermal_domain.add_validator(
            openfoam_solid_validator,
            "thermal_solid_data",
        )
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. Please download the additional files from NGC https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_sym_examples_supplemental_materials"
        )

    # make solver
    thermal_slv = Solver(cfg, thermal_domain)

    # start thermal solver
    thermal_slv.solve()


if __name__ == "__main__":
    run()
