# Copyright (c) Facebook, Inc. and its affiliates.
import os
import pickle

import torch
from torch import nn

from pythia.modules.layers import Identity
from pythia.utils.general import get_pythia_root


class ImageEncoder(nn.Module):
    def __init__(self, encoder_type, in_dim, **kwargs):
        super(ImageEncoder, self).__init__()

        if encoder_type == "default":
            self.module = Identity()
            self.module.in_dim = in_dim
            self.module.out_dim = in_dim
        elif encoder_type == "finetune_faster_rcnn_fpn_fc7":
            self.module = FinetuneFasterRcnnFpnFc7(in_dim, **kwargs)
        else:
            raise NotImplementedError("Unknown Image Encoder: %s" % encoder_type)

        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class FinetuneFasterRcnnFpnFc7(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file, model_data_dir):
        super(FinetuneFasterRcnnFpnFc7, self).__init__()
        pythia_root = get_pythia_root()
        model_data_dir = os.path.join(pythia_root, model_data_dir)

        if not os.path.isabs(weights_file):
            weights_file = os.path.join(model_data_dir, weights_file)
        if not os.path.isabs(bias_file):
            bias_file = os.path.join(model_data_dir, bias_file)
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
        i3 = nn.functional.relu(i2)
        return i3
