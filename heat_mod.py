# example file to model 3d heat equation of the laser heating
# it is modelled based on the example chip_2d in modulus system.

import os
import warnings

import torch
import numpy as np

from sympy import Symbol, Eq, Or, And

import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.domain import Domain
from modulus.sym.solver import Solver
from modulus.sym.geometry import Bounds
from modulus.sym.eq.pdes.navier_stokes import GradNormal
from modulus.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from modulus.sym.geometry.primitives_3d import Box, Channel

# constraints
from modulus.sym.domain.constraint import(
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from modulus.sym.models.activation import Activation
from modulus.sym.domain.monitor import PointwiseMonitor
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.models.modified_fourier_net import ModifiedFourierNetArch

@modulus.sym.main(config_path="heat_3d", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # add constraint to solver
    box_origin = (0.0, 0.0, 0.0)
    box_dim = (2,2,0.24)
    source_origin = (0.8, 0.8, 0.24)
    source_dim = (0.4,0.4,0.0)
    chann_origin = (0.0, 0.0, 0.0)
    chann_dim = (2.0, 2.0, 0.24)

    # initial temperature
    init_temp = 25
    source_grad = 0.025
    diff = 1.0

    # make list of nodes to unroll graph on
    solid_3d = Diffusion(T="theta", D = diff, dim=3, time=False)

    # define boundary conditions
    gn_solid_3d = GradNormal("theta", dim=3, time=False)

    solid_3d_net = ModifiedFourierNetArch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("theta_star")],
        layer_size=128,
        frequencies=("gaussian", 0.2, 64),
        activation_fn=Activation.TANH,
    )

    # create node structure
    nodes = (
        solid_3d.make_nodes()
        + gn_solid_3d.make_nodes()
        + [
            Node.from_sympy(298 * Symbol("theta"), "theta")
        ] # Normalize the outputs
        + [solid_3d_net.make_node(name="solid_3d_network")]
    )

    # define sympy variables to parameterize domain curves
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

    # define geometry

    # define box
    box = Box(
        box_origin,
        (box_origin[0] + box_dim[0], box_origin[1] + box_dim[1], box_origin[2] + box_dim[2])
    )
    channel = Channel(
        chann_origin,
        (chann_origin[0] + chann_dim[0], chann_origin[1] + chann_dim[1], chann_origin[2] + chann_dim[2]),
    )
    print(channel.bounds)
    plate = box
    # make domain
    domain = Domain()

    # walls insulating
    def walls_criteria(invar, params):
        sdf = channel.sdf(invar, params)
        return np.less(sdf["sdf"], -1e-5)

    walls = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry=channel,
        outvar={"normal_gradient_theta": 0},
        batch_size=cfg.batch_size.walls,
        criteria=walls_criteria,
    )
    domain.add_constraint(walls, "insulating_walls")

    # heat source (test)
    heat_source = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry= box,
        outvar={"normal_gradient_theta_source": source_grad},
        batch_size = cfg.batch_size.heat_source,
        lambda_weighting={"normal_gradient_theta_source": 1000},
        criteria =(
            Eq(z, source_origin[2])
            & (y >= source_origin[1])
            & (y <= (source_origin[1] + source_dim[1]))
            & (x >= source_origin[0])
            & (x <= (source_origin[0] + source_dim[0]))
        )
    )
    domain.add_constraint(heat_source, name="heat_source")

    # add monitor of max. temperature
    monitor = PointwiseMonitor(
        plate.sample_boundary(10000, criteria=Eq(z, source_origin[2])),
        output_names=["theta_peak"],
        metrics={
            "peak_temp": lambda var: torch.max(var["theta_peak"]),
        },
        nodes=nodes,
    )


    # add validation data if available

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == '__main__':
    run()


