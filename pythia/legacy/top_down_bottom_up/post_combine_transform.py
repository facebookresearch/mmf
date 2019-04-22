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


def build_post_combine_transform(method, par, in_dim):
    if method == "linear_transform":
        return LinearTransform(in_dim, **par)
    elif method == "conv_transform":
        return ConvTransform(in_dim, **par)
    else:
        raise NotImplementedError("unkown post combime transform type %s" % method)


class LinearTransform(nn.Module):
    def __init__(self, in_dim, **kwargs):
        super(LinearTransform, self).__init__()
        self.lc = weight_norm(
            nn.Linear(in_features=in_dim, out_features=kwargs["out_dim"]), dim=None
        )
        self.out_dim = kwargs["out_dim"]

    def forward(self, x):
        return self.lc(x)


class ConvTransform(nn.Module):
    def __init__(self, in_dim, **kwargs):
        super(ConvTransform, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=kwargs["hidden_dim"], kernel_size=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=kwargs["hidden_dim"],
            out_channels=kwargs["out_dim"],
            kernel_size=1,
        )
        self.out_dim = kwargs["out_dim"]

    def forward(self, x):
        if len(x.size()) == 3:  # N x k xdim
            # N x dim x k x 1
            x_reshape = torch.unsqueeze(x.permute(0, 2, 1), 3)
        elif len(x.size()) == 2:  # N x dim
            # N x dim x 1 x 1
            x_reshape = torch.unsqueeze(torch.unsqueeze(x, 2), 3)

        iatt_conv1 = self.conv1(x_reshape)  # N x hidden_dim x * x 1
        iatt_relu = F.relu(iatt_conv1)
        iatt_conv2 = self.conv2(iatt_relu)  # N x out_dim x * x 1

        if len(x.size()) == 3:
            iatt_conv3 = torch.squeeze(iatt_conv2, 3).permute(0, 2, 1)
        elif len(x.size()) == 2:
            iatt_conv3 = torch.squeeze(torch.squeeze(iatt_conv2, 3), 2)

        return iatt_conv3
