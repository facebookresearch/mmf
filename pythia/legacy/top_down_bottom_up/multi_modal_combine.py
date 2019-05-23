# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import torch.nn.functional as F

from top_down_bottom_up.nonlinear_layer import nonlinear_layer


def build_modal_combine_module(method, par, image_feat_dim, ques_emb_dim):
    if method == "MFH":
        return MFH(image_feat_dim, ques_emb_dim, **par)
    elif method == "non_linear_elmt_multiply":
        return non_linear_elmt_multiply(image_feat_dim, ques_emb_dim, **par)
    elif method == "two_layer_elmt_multiply":
        return two_layer_elmt_multiply(image_feat_dim, ques_emb_dim, **par)
    else:
        raise NotImplemented("unimplemented %s for modal combine module" % method)


class MfbExpand(nn.Module):
    def __init__(self, **kwargs):
        super(MfbExpand, self).__init__()
        self.lc_image = nn.Linear(
            in_features=kwargs["image_feat_dim"], out_features=kwargs["hidden_size"]
        )
        self.lc_ques = nn.Linear(
            in_features=kwargs["ques_emb_dim"], out_features=kwargs["hidden_size"]
        )
        self.dropout = nn.Dropout(kwargs["dropout"])

    def forward(self, image_feat, question_embed):
        image1 = self.lc_image(image_feat)
        ques1 = self.lc_ques(question_embed)
        if len(image_feat.data.shape) == 3:
            num_location = image_feat.data.size(1)
            ques1_expand = torch.unsqueeze(ques1, 1).expand(-1, num_location, -1)
        else:
            ques1_expand = ques1
        joint_feature = image1 * ques1_expand
        joint_feature = self.dropout(joint_feature)
        return joint_feature


# joint_feature dim: N x k x dim or N x dim
class MfbSqueeze(nn.Module):
    def __init__(self, **kwargs):
        super(MfbSqueeze, self).__init__()
        self.pool_size = kwargs["pool_size"]

    def forward(self, joint_feature):

        orig_feature_size = len(joint_feature.size())
        if orig_feature_size == 2:
            joint_feature = torch.unsqueeze(joint_feature, dim=1)
        batch_size, num_loc, dim = joint_feature.size()

        if dim % self.pool_size != 0:
            exit(
                "the dim %d is not multiply of \
             pool_size %d"
                % (dim, self.pool_size)
            )
        joint_feature_reshape = joint_feature.view(
            batch_size, num_loc, int(dim / self.pool_size), self.pool_size
        )
        # N x 100 x 1000 x 1
        iatt_iq_sumpool = torch.sum(joint_feature_reshape, 3)
        iatt_iq_sqrt = torch.sqrt(F.relu(iatt_iq_sumpool)) - torch.sqrt(
            F.relu(-iatt_iq_sumpool)
        )
        iatt_iq_sqrt = iatt_iq_sqrt.view(batch_size, -1)  # N x 100000
        iatt_iq_l2 = F.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(batch_size, num_loc, int(dim / self.pool_size))
        if orig_feature_size == 2:
            iatt_iq_l2 = torch.squeeze(iatt_iq_l2, dim=1)

        return iatt_iq_l2


class MFH(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(MFH, self).__init__()
        self.order = kwargs["order"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.mfb_expand_list = nn.ModuleList()
        self.mfb_sqz_list = nn.ModuleList()
        self.out_dim = int(sum(hidden_sizes) / kwargs["pool_size"])
        for i in range(self.order):
            mfb_exp_i = MfbExpand(
                image_feat_dim=image_feat_dim,
                ques_emb_dim=ques_emb_dim,
                hidden_size=hidden_sizes[i],
                dropout=kwargs["dropout"],
            )
            self.mfb_expand_list.append(mfb_exp_i)
            mfb_sqz_i = MfbSqueeze(pool_size=kwargs["pool_size"])
            self.mfb_sqz_list.append(mfb_sqz_i)

    def forward(self, image_feat, question_embedding):
        feature_list = []
        prev_mfb_exp = 1
        for i in range(self.order):
            mfb_exp = self.mfb_expand_list[i]
            mfb_sqz = self.mfb_sqz_list[i]
            z_exp_i = mfb_exp(image_feat, question_embedding)
            if i > 0:
                z_exp_i = prev_mfb_exp * z_exp_i
            prev_mfb_exp = z_exp_i
            z = mfb_sqz(z_exp_i)
            feature_list.append(z)

        # append at last feature
        cat_dim = len(feature_list[0].size()) - 1
        feature = torch.cat(feature_list, dim=cat_dim)
        return feature


# need to handle two situations,
# first: image (N, K, i_dim), question (N, q_dim);
# second: image (N, i_dim), question (N, q_dim);
class non_linear_elmt_multiply(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(non_linear_elmt_multiply, self).__init__()
        self.Fa_image = nonlinear_layer(image_feat_dim, kwargs["hidden_size"])
        self.Fa_txt = nonlinear_layer(ques_emb_dim, kwargs["hidden_size"])
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.out_dim = kwargs["hidden_size"]

    def forward(self, image_feat, question_embedding):
        image_fa = self.Fa_image(image_feat)
        question_fa = self.Fa_txt(question_embedding)

        if len(image_feat.data.shape) == 3:
            num_location = image_feat.data.size(1)
            question_fa_expand = torch.unsqueeze(question_fa, 1).expand(
                -1, num_location, -1
            )
        else:
            question_fa_expand = question_fa

        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)

        return joint_feature


class two_layer_elmt_multiply(nn.Module):
    def __init__(self, image_feat_dim, ques_emb_dim, **kwargs):
        super(two_layer_elmt_multiply, self).__init__()
        self.Fa_image1 = nonlinear_layer(image_feat_dim, kwargs["hidden_size"])
        self.Fa_image2 = nonlinear_layer(kwargs["hidden_size"], kwargs["hidden_size"])
        self.Fa_txt1 = nonlinear_layer(ques_emb_dim, kwargs["hidden_size"])
        self.Fa_txt2 = nonlinear_layer(kwargs["hidden_size"], kwargs["hidden_size"])
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.out_dim = kwargs["hidden_size"]

    def forward(self, image_feat, question_embedding):
        image_fa = self.Fa_image2(self.Fa_image1(image_feat))
        question_fa = self.Fa_txt2(self.Fa_txt1(question_embedding))

        if len(image_feat.data.shape) == 3:
            num_location = image_feat.data.size(1)
            question_fa_expand = torch.unsqueeze(question_fa, 1).expand(
                -1, num_location, -1
            )
        else:
            question_fa_expand = question_fa

        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)

        return joint_feature
