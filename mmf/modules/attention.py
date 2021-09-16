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


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs
