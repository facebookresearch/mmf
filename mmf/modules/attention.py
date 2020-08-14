# Copyright (c) Facebook, Inc. and its affiliates.

import math
from typing import Optional, Tuple, Type

import torch
from mmf.modules.layers import GatedTanh, ModalCombineLayer, TransformLayer
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, image_dim, question_dim, **kwargs):
        super().__init__()

        combine_type = kwargs["modal_combine"]["type"]
        combine_params = kwargs["modal_combine"]["params"]
        modal_combine_layer = ModalCombineLayer(
            combine_type, image_dim, question_dim, **combine_params
        )

        transform_type = kwargs["transform"]["type"]
        transform_params = kwargs["transform"]["params"]
        transform_layer = TransformLayer(
            transform_type, modal_combine_layer.out_dim, **transform_params
        )

        normalization = kwargs["normalization"]

        self.module = TopDownAttention(
            modal_combine_layer, transform_layer, normalization
        )

        if hasattr(self.module, "out_dim"):
            self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ConcatenationAttention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size):
        super().__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.fa = GatedTanh(image_feat_dim + txt_rnn_embeding_dim, hidden_size)
        self.lc = nn.Linear(hidden_size, 1)

    def forward(self, image_feat, question_embedding):
        _, num_location, _ = image_feat.shape
        question_embedding_expand = torch.unsqueeze(question_embedding, 1).expand(
            -1, num_location, -1
        )
        concat_feature = torch.cat((image_feat, question_embedding_expand), dim=2)
        raw_attention = self.lc(self.fa(concat_feature))
        # softmax across locations
        attention_weights = nn.functional.softmax(raw_attention, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class ProjectAttention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout=0.2):
        super().__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.fa_image = GatedTanh(image_feat_dim, hidden_size)
        self.fa_txt = GatedTanh(txt_rnn_embeding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lc = nn.Linear(hidden_size, 1)

    def compute_raw_att(self, image_feat, question_embedding):
        num_location = image_feat.shape[1]
        image_fa = self.fa_image(image_feat)
        question_fa = self.fa_txt(question_embedding)
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
        attention_weights = nn.functional.softmax(raw_attention, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class DoubleProjectAttention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout=0.2):
        super().__init__()
        self.att1 = ProjectAttention(
            image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout
        )
        self.att2 = ProjectAttention(
            image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout
        )
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim

    def forward(self, image_feat, question_embedding):
        att1 = self.att1.compute_raw_att(image_feat, question_embedding)
        att2 = self.att2.compute_raw_att(image_feat, question_embedding)
        raw_attn_weights = att1 + att2
        # softmax across locations
        attention_weights = nn.functional.softmax(raw_attn_weights, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class TopDownAttention(nn.Module):
    EPS = 1.0e-08

    def __init__(self, combination_layer, transform_module, normalization):
        super().__init__()
        self.combination_layer = combination_layer
        self.normalization = normalization
        self.transform = transform_module
        self.out_dim = self.transform.out_dim

    @staticmethod
    def _mask_attentions(attention, image_locs):
        batch_size, num_loc, n_att = attention.size()
        tmp1 = attention.new_zeros(num_loc)
        tmp1[:num_loc] = torch.arange(0, num_loc, dtype=attention.dtype).unsqueeze(
            dim=0
        )

        tmp1 = tmp1.expand(batch_size, num_loc)
        tmp2 = image_locs.type(tmp1.type())
        tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
        mask = torch.ge(tmp1, tmp2)
        mask = mask.unsqueeze(dim=2).expand_as(attention)
        attention = attention.masked_fill(mask, 0)
        return attention

    def forward(self, image_feat, question_embedding, image_locs=None):
        # N x K x joint_dim
        joint_feature = self.combination_layer(image_feat, question_embedding)
        # N x K x n_att
        raw_attn = self.transform(joint_feature)

        if self.normalization.lower() == "softmax":
            attention = nn.functional.softmax(raw_attn, dim=1)
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)
                masked_attention_sum = torch.sum(masked_attention, dim=1, keepdim=True)
                masked_attention_sum += masked_attention_sum.eq(0).float() + self.EPS
                masked_attention = masked_attention / masked_attention_sum
            else:
                masked_attention = attention

        elif self.normalization.lower() == "sigmoid":
            attention = torch.sigmoid(raw_attn)
            masked_attention = attention
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)

        return masked_attention


# TODO(vedanuj): Remove this and use torch.nn.MultiHeadAttention
class MovieMcanMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementation from https://arxiv.org/abs/1706.03762
    used for Movie+MCAN
    """

    def __init__(self, dim: int, num_attn: int, dropout: float = 0.1):
        super().__init__()
        self.p_attn = None
        self.h = num_attn
        self.d_k = dim // num_attn
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def qkv_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Type[nn.Dropout] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores.data.masked_fill_(mask.unsqueeze(1).unsqueeze(2), -1e9)

        p_attn = nn.functional.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        b = q.size(0)

        q = self.linears[0](q).view(b, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linears[1](k).view(b, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linears[2](v).view(b, -1, self.h, self.d_k).transpose(1, 2)

        x, self.p_attn = self.qkv_attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)

        return self.linears[-1](x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_attn: int, dropout: float):
        super().__init__()
        self.multi_head_attn = MovieMcanMultiHeadAttention(dim, num_attn, dropout=0.1)
        self.fcn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4 * dim, dim),
        )
        self.drop_mha = nn.Dropout(p=dropout)
        self.ln_mha = nn.LayerNorm(dim)
        self.drop_fcn = nn.Dropout(p=dropout)
        self.ln_fcn = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.ln_mha(x + self.drop_mha(self.multi_head_attn(x, x, x, x_mask)))
        x = self.ln_fcn(x + self.drop_fcn(self.fcn(x)))

        return x


class SelfGuidedAttention(nn.Module):
    def __init__(self, dim: int, num_attn: int, dropout: float):
        super().__init__()
        self.multi_head_attn = nn.ModuleList(
            [MovieMcanMultiHeadAttention(dim, num_attn, dropout=0.1) for _ in range(2)]
        )
        self.fcn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4 * dim, dim),
        )
        self.drop_mha = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])
        self.ln_mha = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.drop_fcn = nn.Dropout(p=dropout)
        self.ln_fcn = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.ln_mha[0](
            x + self.drop_mha[0](self.multi_head_attn[0](x, x, x, x_mask))
        )
        x = self.ln_mha[1](
            x + self.drop_mha[1](self.multi_head_attn[1](x, y, y, y_mask))
        )
        x = self.ln_fcn(x + self.drop_fcn(self.fcn(x)))

        return x
