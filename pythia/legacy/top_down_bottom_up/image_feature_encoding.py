# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg


def build_image_feature_encoding(method, par, in_dim):
    if method == "default_image":
        return DefaultImageFeature(in_dim)
    elif method == "finetune_faster_rcnn_fpn_fc7":
        return FinetuneFasterRcnnFpnFc7(in_dim, **par)
    else:
        raise NotImplementedError("unknown image feature encoding %s" % method)


class DefaultImageFeature(nn.Module):
    def __init__(self, in_dim):
        super(DefaultImageFeature, self).__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, image):
        return image


class FinetuneFasterRcnnFpnFc7(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file):
        super(FinetuneFasterRcnnFpnFc7, self).__init__()
        if not os.path.isabs(weights_file):
            weights_file = os.path.join(cfg.data.data_root_dir, weights_file)
        if not os.path.isabs(bias_file):
            bias_file = os.path.join(cfg.data.data_root_dir, bias_file)
        with open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = F.relu(i2)
        return i3
