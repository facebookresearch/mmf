# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from top_down_bottom_up.nonlinear_layer import nonlinear_layer


def build_classifier(method, par, in_dim, out_dim):
    classifier_par = par
    if method == "weight_norm_classifier":
        return WeightNormClassifier(in_dim, out_dim, **classifier_par)
    elif method == "logit_classifier":
        return logit_classifier(in_dim, out_dim, **classifier_par)
    elif method == "linear_classifier":
        return LinearClassifier(in_dim, out_dim, **classifier_par)
    else:
        raise NotImplementedError("unknown classifier %s" % method)


class logit_classifier(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(logit_classifier, self).__init__()
        input_dim = in_dim
        num_ans_candidates = out_dim
        txt_nonLinear_dim = kwargs["txt_hidden_dim"]
        image_nonLinear_dim = kwargs["img_hidden_dim"]
        self.f_o_text = nonlinear_layer(input_dim, txt_nonLinear_dim)
        self.f_o_image = nonlinear_layer(input_dim, image_nonLinear_dim)
        self.linear_text = nn.Linear(txt_nonLinear_dim, num_ans_candidates)
        self.linear_image = nn.Linear(image_nonLinear_dim, num_ans_candidates)
        if "pretrained_image" in kwargs and kwargs["pretrained_text"] is not None:
            self.linear_text.weight.data.copy_(
                torch.from_numpy(kwargs["pretrained_text"])
            )
        if "pretrained_image" in kwargs and kwargs["pretrained_image"] is not None:
            self.linear_image.weight.data.copy_(
                torch.from_numpy(kwargs["pretrained_image"])
            )

    def forward(self, joint_embedding):
        text_val = self.linear_text(self.f_o_text(joint_embedding))
        image_val = self.linear_image(self.f_o_image(joint_embedding))
        logit_value = text_val + image_val

        return logit_value


class WeightNormClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(WeightNormClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, kwargs["hidden_dim"]), dim=None),
            nn.ReLU(),
            nn.Dropout(kwargs["dropout"], inplace=True),
            weight_norm(nn.Linear(kwargs["hidden_dim"], out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearClassifier, self).__init__()
        self.lc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lc(x)
