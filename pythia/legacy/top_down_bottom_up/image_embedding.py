# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    parameters:

    input:
        image_feat_variable: [batch_size, num_location, image_feat_dim]
            or a list of [num_location, image_feat_dim]
            when using adaptive number of objects
        question_embedding:[batch_size, txt_embeding_dim]

    output:
        image_embedding:[batch_size, image_feat_dim]


"""


class image_embedding(nn.Module):
    def __init__(self, image_attention_model):
        super(image_embedding, self).__init__()
        self.image_attention_model = image_attention_model
        self.out_dim = image_attention_model.out_dim

    def forward(self, image_feat_variable, question_embedding, image_dims):
        # N x K x n_att
        attention = self.image_attention_model(
            image_feat_variable, question_embedding, image_dims
        )
        att_reshape = attention.permute(0, 2, 1)
        tmp_embedding = torch.bmm(
            att_reshape, image_feat_variable
        )  # N x n_att x image_dim
        batch_size = att_reshape.size(0)
        image_embedding = tmp_embedding.view(batch_size, -1)

        return image_embedding


class image_finetune(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file):
        super(image_finetune, self).__init__()
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
