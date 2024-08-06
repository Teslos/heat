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

import torch
from torch.utils.data import DataLoader, Dataset

from sympy import Symbol, Eq, Abs, tanh

import numpy as np

from modulus.sym.utils.io import csv_to_dict
from modulus.sym.solver import Solver
from modulus.sym.geometry import Parameterization
from modulus.sym.geometry.primitives_3d import Box, Channel, Plane
from modulus.sym.key import Key
from modulus.sym.node import Node

# define sympy varaibles to parametize domain curves
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
x_pos = Symbol("x_pos")
fin_length_s = 1.0
fin_height_s = 0.4
fin_thickness_s = 0.6

# geometry params for domain
channel_origin = (-2.5, -0.5, -0.5)
channel_dim = (5.0, 1.0, 1.0)
heat_sink_base_origin = (-1.0, -0.5, -0.3)
heat_sink_base_dim = (1.0, 0.2, 0.6)
fin_origin = (heat_sink_base_origin[0] + 0.5 - fin_length_s / 2, -0.3, -0.3)
fin_dim = (fin_length_s, fin_height_s, fin_thickness_s)  # two side fins
flow_box_origin = (-1.1, -0.5, -0.5)
flow_box_dim = (1.6, 1.0, 1.0)
source_origin = (-0.7, -0.5, -0.1)
source_dim = (0.4, 0.0, 0.2)
source_area = 0.08


# define geometry
class ThreeFin(object):
    def __init__(self):
       

        # channel
        self.channel = Channel(
            channel_origin,
            (
                channel_origin[0] + channel_dim[0],
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
        )

        # three fin heat sink
        heat_sink_base = Box(
            heat_sink_base_origin,
            (
                heat_sink_base_origin[0] + heat_sink_base_dim[0],  # base of heat sink
                heat_sink_base_origin[1] + heat_sink_base_dim[1],
                heat_sink_base_origin[2] + heat_sink_base_dim[2],
            ),
        )
        fin_center = (
            fin_origin[0] + fin_dim[0] / 2,
            fin_origin[1] + fin_dim[1] / 2,
            fin_origin[2] + fin_dim[2] / 2,
        )
        fin = Box(
            fin_origin,
            (
                fin_origin[0] + fin_dim[0],
                fin_origin[1] + fin_dim[1],
                fin_origin[2] + fin_dim[2],
            ),
        )
       
        three_fin = heat_sink_base + fin

      
        self.three_fin = three_fin 

        # entire geometry
        self.geo = self.channel - self.three_fin

        # low and high resultion geo away and near the heat sink
        flow_box = Box(
            flow_box_origin,
            (
                flow_box_origin[0] + flow_box_dim[0],  # base of heat sink
                flow_box_origin[1] + flow_box_dim[1],
                flow_box_origin[2] + flow_box_dim[2],
            ),
        )
        self.lr_geo = self.geo - flow_box
        self.hr_geo = self.geo & flow_box
        lr_bounds_x = (channel_origin[0], channel_origin[0] + channel_dim[0])
        lr_bounds_y = (channel_origin[1], channel_origin[1] + channel_dim[1])
        lr_bounds_z = (channel_origin[2], channel_origin[2] + channel_dim[2])
        self.lr_bounds = {x: lr_bounds_x, y: lr_bounds_y, z: lr_bounds_z}
        hr_bounds_x = (flow_box_origin[0], flow_box_origin[0] + flow_box_dim[0])
        hr_bounds_y = (flow_box_origin[1], flow_box_origin[1] + flow_box_dim[1])
        hr_bounds_z = (flow_box_origin[2], flow_box_origin[2] + flow_box_dim[2])
        self.hr_bounds = {x: hr_bounds_x, y: hr_bounds_y, z: hr_bounds_z}

        # inlet and outlet
        self.inlet = Plane(
            channel_origin,
            (
                channel_origin[0],
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
            -1,
        )
        self.outlet = Plane(
            (channel_origin[0] + channel_dim[0], channel_origin[1], channel_origin[2]),
            (
                channel_origin[0] + channel_dim[0],
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
            1,
        )

        # planes for integral continuity
        self.integral_plane = Plane(
            (x_pos, channel_origin[1], channel_origin[2]),
            (
                x_pos,
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
            1,
        )
