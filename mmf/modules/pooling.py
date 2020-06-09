# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool1d(nn.Module):
    """
    An attention pooling function that learns weights using an mlp  
    """
    def __init__(self, num_features, num_attn=1, dropout=0.1):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(num_features // 2, num_attn),
        )
        self.p_attn = None
        self.num_attn = num_attn

    def forward(self, query, value, mask=None):
        b = query.size(0)
        score = self.linear(query).transpose(-2, -1)
        if mask is not None:
            score.data.masked_fill_(mask.unsqueeze(1), -1e9)
        self.p_attn = F.softmax(score, dim=-1)

        return torch.matmul(self.p_attn, value).view(b, self.num_attn, -1)


class BranchCombineLayer(nn.Module):
    """
    Three-branch fusion module which fuses MoVie and MCAN in
    https://arxiv.org/abs/2004.11883
    """
    def __init__(self, img_dim, ques_dim):
        super().__init__()
        self.out_dim = img_dim * 2
        self.linear_cga = nn.ModuleList(
            [nn.Linear(img_dim, self.out_dim) for _ in range(2)]
        )
        self.linear_cbn = nn.ModuleList(
            [nn.Linear(1024, self.out_dim) for _ in range(2)]
        )
        self.linear_ques = nn.ModuleList(
            [nn.Linear(ques_dim, self.out_dim) for _ in range(2)]
        )
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.out_dim) for _ in range(3)]
        )

    def forward(self, v_cga, v_cbn, q):
        feat = [
            self.layer_norm[0](
                self.linear_ques[0](q)
                + self.linear_cbn[0](v_cbn)
                + self.linear_cga[0](v_cga)
            ),
            self.layer_norm[1](self.linear_cbn[1](v_cbn)),
            self.layer_norm[2](self.linear_ques[1](q) + self.linear_cga[1](v_cga)),
        ]

        if self.training:
            return torch.stack(feat, dim=1)

        return feat[0]
