# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import torch.nn.functional as F

from global_variables.global_variables import use_cuda
from top_down_bottom_up.multi_modal_combine import build_modal_combine_module
from top_down_bottom_up.nonlinear_layer import nonlinear_layer
from top_down_bottom_up.post_combine_transform import \
    build_post_combine_transform


class concatenate_attention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size):
        super(concatenate_attention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.Fa = nonlinear_layer(image_feat_dim + txt_rnn_embeding_dim, hidden_size)
        self.lc = nn.Linear(hidden_size, 1)

    def forward(self, image_feat, question_embedding):
        _, num_location, _ = image_feat.shape
        question_embedding_expand = torch.unsqueeze(question_embedding, 1).expand(
            -1, num_location, -1
        )
        concat_feature = torch.cat((image_feat, question_embedding_expand), dim=2)
        raw_attention = self.lc(self.Fa(concat_feature))
        # softmax across locations
        attention = F.softmax(raw_attention, dim=1).expand_as(image_feat)
        return attention


class project_attention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout=0.2):
        super(project_attention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.Fa_image = nonlinear_layer(image_feat_dim, hidden_size)
        self.Fa_txt = nonlinear_layer(txt_rnn_embeding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lc = nn.Linear(hidden_size, 1)

    def compute_raw_att(self, image_feat, question_embedding):
        _, num_location, _ = image_feat.shape
        image_fa = self.Fa_image(image_feat)
        question_fa = self.Fa_txt(question_embedding)
        question_fa_expand = torch.unsqueeze(question_fa, 1).expand(
            -1, num_location, -1
        )
        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)
        raw_attention = self.lc(joint_feature)
        return raw_attention

    def forward(self, image_feat, question_embedding):
        raw_attention = self.compute_raw_att(image_feat, question_embedding)
        # softmax across locations
        attention = F.softmax(raw_attention, dim=1).expand_as(image_feat)
        return attention


class doubel_project_attention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout=0.2):
        super(doubel_project_attention, self).__init__()
        self.att1 = project_attention(
            image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout
        )
        self.att2 = project_attention(
            image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout
        )
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim

    def forward(self, image_feat, question_embedding):
        att1 = self.att1.compute_raw_att(image_feat, question_embedding)
        att2 = self.att2.compute_raw_att(image_feat, question_embedding)
        raw_attention = att1 + att2
        # softmax across locations
        attention = F.softmax(raw_attention, dim=1).expand_as(image_feat)
        return attention


def build_image_attention_module(image_att_model_par, image_dim, ques_dim):
    modal_combine_module = build_modal_combine_module(
        image_att_model_par["modal_combine"]["method"],
        image_att_model_par["modal_combine"]["par"],
        image_feat_dim=image_dim,
        ques_emb_dim=ques_dim,
    )

    transform_module = build_post_combine_transform(
        image_att_model_par["transform"]["method"],
        image_att_model_par["transform"]["par"],
        in_dim=modal_combine_module.out_dim,
    )

    normalization = image_att_model_par["normalization"]

    return top_down_attention(modal_combine_module, normalization, transform_module)


"""
question_embedding: N x q_dim

"""


class top_down_attention(nn.Module):
    def __init__(self, modal_combine_module, normalization, transform_module):
        super(top_down_attention, self).__init__()
        self.modal_combine = modal_combine_module
        self.normalization = normalization
        self.transform = transform_module
        self.out_dim = self.transform.out_dim

    @staticmethod
    def _mask_attentions(attention, image_locs):
        batch_size, num_loc, n_att = attention.data.shape
        tmp1 = torch.unsqueeze(
            torch.arange(0, num_loc).type(torch.LongTensor), dim=0
        ).expand(batch_size, num_loc)
        tmp1 = tmp1.cuda() if use_cuda else tmp1
        tmp2 = torch.unsqueeze(image_locs.data, 1).expand(batch_size, num_loc)
        mask = torch.ge(tmp1, tmp2)
        mask = torch.unsqueeze(mask, 2).expand_as(attention)
        attention.data.masked_fill_(mask, 0)
        return attention

    def forward(self, image_feat, question_embedding, image_locs=None):
        # N x K x joint_dim
        joint_feature = self.modal_combine(image_feat, question_embedding)
        # N x K x n_att
        raw_attention = self.transform(joint_feature)

        if self.normalization.lower() == "softmax":
            attention = F.softmax(raw_attention, dim=1)
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)
                masked_attention_sum = torch.sum(masked_attention, dim=1, keepdim=True)
                masked_attention = masked_attention / masked_attention_sum
            else:
                masked_attention = attention

        elif self.normalization.lower() == "sigmoid":
            attention = F.sigmoid(raw_attention)
            masked_attention = attention
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)

        return masked_attention
