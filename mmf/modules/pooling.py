# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, dim, elementwise_affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        b, c, h, w = x.shape

        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class PositionEncoding(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.pe = nn.Parameter(nn.init.xavier_normal_(torch.empty(*shape)))

    def forward(self, input):
        assert input.dim() == 4, "Tensor of 4-dim is required"
        return input + self.pe[:, :, : input.shape[2], : input.shape[3]]


class AttnPool1d(nn.Module):
    def __init__(self, num_features, num_attn=1, dropout=0.1, mode="mlp"):
        super().__init__()
        if mode == "mlp":
            self.linear = nn.Sequential(
                nn.Linear(num_features, num_features // 2),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(num_features // 2, num_attn),
            )
        elif mode == "linear":
            self.linear = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(num_features, num_attn),
            )
        else:
            raise ValueError(f"Only 'linear' and 'mlp' supported! Found {mode}")
        self.p_attn = None
        self.num_attn = num_attn

    def forward(self, query, value, mask=None):
        # query: B x L x C
        # print("query:", query)
        b = query.size(0)
        score = self.linear(query).transpose(-2, -1)
        if mask is not None:
            score.data.masked_fill_(mask.unsqueeze(1), -1e9)

        self.p_attn = F.softmax(score, dim=-1)

        return torch.matmul(self.p_attn, value).view(b, self.num_attn, -1)


class AdaptiveSumPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sum(3).sum(2)


class ContextPool2d(nn.Module):
    def __init__(self, img_dim, ques_dim, score_type="multi"):
        super().__init__()
        self.score_type = score_type
        if score_type == "cbn":
            self.linear_scale = nn.Linear(ques_dim, img_dim)
            self.linear_bias = nn.Linear(ques_dim, img_dim)
        else:
            self.linear_ques = nn.Linear(ques_dim, img_dim)

    def forward(self, v, q, c=None):
        b, c, h, w = v.shape
        if self.score_type == "multi":
            score = (v * self.linear_ques(q).unsqueeze(2).unsqueeze(3)).sum(
                1, keepdim=True
            )
        elif self.score_type == "sum":
            score = (v + self.linear_ques(q).unsqueeze(2).unsqueeze(3)).sum(
                1, keepdim=True
            )
        elif self.score_type == "cbn":
            score = (
                v * self.linear_scale(q).unsqueeze(2).unsqueeze(3)
                + self.linear_bias(c).unsqueeze(2).unsqueeze(3)
            ).sum(1, keepdim=True)
        score = F.softmax(score.view(b, 1, h * w), dim=-1).view(b, 1, h, w)
        return (v * score).sum(3).sum(2)


class FusionPool2d(nn.Module):
    def __init__(self, img_dim, ques_dim, fusion_type="multi"):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "cat":
            self.linear_ques = nn.Linear(ques_dim, ques_dim)
        elif fusion_type == "cbn":
            self.linear_ques = nn.Linear(ques_dim, img_dim)
            self.linear = nn.Linear(img_dim, img_dim)
            self.layer_norm = nn.LayerNorm(img_dim)
        elif fusion_type == "mcan":
            self.out_dim = img_dim * 2
            self.ques_pool = AttnPool1d(ques_dim, 1)
            self.img_pool = AttnPool1d(img_dim, 1)
            self.linear_ques = nn.Linear(ques_dim, self.out_dim)
            self.linear_img = nn.Linear(img_dim, self.out_dim)
            self.layer_norm = nn.LayerNorm(self.out_dim)
        elif fusion_type == "multi":
            self.linear_ques = nn.Linear(ques_dim, img_dim)
        elif fusion_type == "flatten_multi":
            self.linear_ques = nn.Linear(ques_dim, img_dim)
        elif fusion_type == "flatten_cbn":
            self.linear_ques = nn.Linear(ques_dim, img_dim)
            self.conv = nn.Conv2d(img_dim, img_dim, kernel_size=1)

    def forward(self, v, q, v_mask=None, q_mask=None):
        if self.fusion_type == "cat":
            feat = torch.cat([v, self.linear_ques(q).squeeze(3).squeeze(2)], dim=1)
        elif self.fusion_type == "cbn":
            feat = self.layer_norm(self.linear(v * self.linear_ques(q)) + v)
        elif self.fusion_type == "mcan":
            v_pool = self.linear_img(self.img_pool(v, v, v_mask).squeeze(1))
            q_pool = self.linear_ques(self.ques_pool(q, q, q_mask).squeeze(1))
            feat = self.layer_norm(v_pool + q_pool)
        elif self.fusion_type == "multi":
            feat = self.linear_ques(q) * v
        elif self.fusion_type == "flatten_multi":
            assert v.dim() == 4, "visual feature dimension should be 4!"
            b, c, h, w = v.shape
            ques = self.linear_ques(q).unsqueeze(2).unsqueeze(3)
            feat = (ques * v).view(b, -1)
        elif self.fusion_type == "flatten_cbn":
            assert v.dim() == 4, "visual feature dimension should be 4!"
            b, c, h, w = v.shape
            ques = self.linear_ques(q).unsqueeze(2).unsqueeze(3)
            feat = (self.conv(ques * v) + v).view(b, -1)

        return feat


class MultiscaleFusion(nn.Module):
    def __init__(self, img_dim, ques_dim, fusion_type):
        super().__init__()
        if fusion_type == "attn":
            self.linear = nn.Conv2d(ques_dim, img_dim, kernel_size=1)
        self.fusion_type = fusion_type

    def forward(self, v, q):
        # v: b x num_branch x c x h x w
        if self.fusion_type == "attn":
            q = self.linear(q.unsqueeze(2).unsqueeze(3)).unsqueeze(
                1
            )  # q: b x 1 x c x 1 x 1
            score = F.softmax(
                (q * v).sum(dim=2, keepdim=True), dim=1
            )  # b x num_branch x 1 x h x w

            return (v * score).sum(dim=1)

        return v.sum(dim=1)


class Softmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)


class CombineLayer(nn.Module):
    def __init__(self, img_dim, ques_dim, fusion_type):
        super().__init__()
        self.linear_cga = nn.Linear(img_dim, 1024)
        self.linear_cbn = nn.Linear(2048, 1024)
        self.linear_ques = nn.Linear(ques_dim, 1024)
        self.layer_norm = nn.LayerNorm(1024)
        self.fusion_type = fusion_type
        self.out_dim = 1024
        if fusion_type in ["gate", "sep_gate"]:
            self.gate = nn.Sequential(nn.Linear(ques_dim, 2), nn.Sigmoid(),)
        elif fusion_type in ["softmax", "sep_softmax"]:
            self.gate = nn.Sequential(nn.Linear(ques_dim, 2), Softmax(dim=-1),)

    def forward(self, v_cga, v_cbn, q):
        if self.fusion_type == "sum":
            feat = self.layer_norm(
                self.linear_ques(q) + self.linear_cbn(v_cbn) + self.linear_cga(v_cga)
            )
        elif self.fusion_type in ["gate", "softmax"]:
            gate = self.gate(q)
            feat = self.layer_norm(
                self.linear_ques(q)
                + self.linear_cbn(v_cbn) * gate[:, 0].unsqueeze(1)
                + self.linear_cga(v_cga) * gate[:, 1].unsqueeze(1)
            )
        elif self.fusion_type in ["sep_gate", "sep_softmax"]:
            gate = self.gate(q)
            feat = self.layer_norm(
                self.linear_ques(q) * gate[:, 1].unsqueeze(1)
                + self.linear_cbn(v_cbn) * gate[:, 0].unsqueeze(1)
                + self.linear_cga(v_cga) * gate[:, 1].unsqueeze(1)
            )

        return feat


class BranchCombineLayer(nn.Module):
    def __init__(self, img_dim, ques_dim, fusion_type):
        super().__init__()
        self.out_dim = img_dim * 2
        if "shared" in fusion_type:
            self.linear_cga = nn.Linear(img_dim, self.out_dim)
            self.linear_cbn = nn.Linear(1024, self.out_dim)
            self.linear_ques = nn.Linear(ques_dim, self.out_dim)
            self.layer_norm = nn.ModuleList(
                [nn.LayerNorm(self.out_dim) for _ in range(3)]
            )
        else:
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
        self.fusion_type = fusion_type
        if fusion_type in ["gate", "sep_gate"]:
            self.gate = nn.Sequential(nn.Linear(ques_dim, 2), nn.Sigmoid(),)
        elif fusion_type in ["softmax", "sep_softmax"]:
            self.gate = nn.Sequential(nn.Linear(ques_dim, 2), Softmax(dim=-1),)

    def forward(self, v_cga, v_cbn, q):
        if self.fusion_type == "sum":
            feat = [
                self.layer_norm[0](
                    self.linear_ques[0](q)
                    + self.linear_cbn[0](v_cbn)
                    + self.linear_cga[0](v_cga)
                ),
                self.layer_norm[1](self.linear_cbn[1](v_cbn)),
                # F.relu(self.linear_cbn[1](v_cbn)),
                self.layer_norm[2](self.linear_ques[1](q) + self.linear_cga[1](v_cga)),
            ]
        elif self.fusion_type == "shared_sum":
            feat = [
                self.layer_norm[0](
                    self.linear_ques(q)
                    + self.linear_cbn(v_cbn)
                    + self.linear_cga(v_cga)
                ),
                self.layer_norm[1](self.linear_cbn(v_cbn)),
                self.layer_norm[2](self.linear_ques(q) + self.linear_cga(v_cga)),
            ]
        elif self.fusion_type in ["gate", "softmax"]:
            gate = self.gate(q)
            feat = [
                self.layer_norm[0](
                    self.linear_ques[0](q)
                    + self.linear_cbn[0](v_cbn) * gate[:, 0].unsqueeze(1)
                    + self.linear_cga[0](v_cga) * gate[:, 1].unsqueeze(1)
                ),
                self.layer_norm[1](self.linear_cbn[1](v_cbn)),
                self.layer_norm[2](self.linear_ques[1](q) + self.linear_cga[1](v_cga)),
            ]
        elif self.fusion_type in ["sep_gate", "sep_softmax"]:
            gate = self.gate(q)
            feat = [
                self.layer_norm[0](
                    self.linear_ques[0](q) * gate[:, 1].unsqueeze(1)
                    + self.linear_cbn[0](v_cbn) * gate[:, 0].unsqueeze(1)
                    + self.linear_cga[0](v_cga) * gate[:, 1].unsqueeze(1)
                ),
                self.layer_norm[1](self.linear_cbn[1](v_cbn)),
                self.layer_norm[2](self.linear_ques[1](q) + self.linear_cga[1](v_cga)),
            ]

        if self.training:
            return torch.stack(feat, dim=1)

        return feat[0]
