# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


"""
nonlinear_layer: f_a : x\in R^m => y \in R^n
\tilda{y} = tanh(Wx + b)
g = sigmoid(W'x + b')
y = \tilda(y) \circ g
input (N, *, in_dim)
output (N, *, out_dim)
"""


class nonlinear_layer_org(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(nonlinear_layer_org, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y_tilda = F.tanh(self.fc1(x))
        g = F.sigmoid(self.gate(x))
        y = y_tilda * g
        return y


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """

    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class nonlinear_layer(nn.Module):
    """Simple class for non-linear fully connect network
    """

    def __init__(self, in_dim, out_dim):
        super(nonlinear_layer, self).__init__()

        layers = []
        layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
