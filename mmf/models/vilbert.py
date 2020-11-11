# Copyright (c) Facebook, Inc. and its affiliates.

import math
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.hf_layers import replace_with_jit
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_bert import (
    ACT2FN,
    BertConfig,
    BertEmbeddings,
    BertIntermediate,
    BertLMPredictionHead,
    BertOutput,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
    BertSelfOutput,
)


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.visualization = config.visualization

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in
        # BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
        else:
            attn_data = {}

        return context_layer, attn_data


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self, input_tensor: Tensor, attention_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs

    @torch.no_grad()
    def forward_no_grad(self, hidden_states, attention_mask):
        return self.forward(hidden_states, attention_mask)


class BertImageSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.v_hidden_size, config.v_num_attention_heads)
            )
        self.dynamic_attention = config.dynamic_attention
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(
            config.v_hidden_size / config.v_num_attention_heads
        )

        self.visualization = config.visualization

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)

        if self.dynamic_attention:
            self.dyLinear_q = nn.Linear(config.hidden_size, self.all_head_size)
            self.dyLinear_k = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        txt_embedding: Tensor,
        txt_attention_mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if (
            self.dynamic_attention
            and hasattr(self, "dyLinear_q")
            and hasattr(self, "dyLinear_k")
        ):
            pool_embedding = (txt_embedding * txt_attention_mask).sum(1)
            pool_embedding = pool_embedding / txt_attention_mask.sum(1)

            # given pool embedding, Linear and Sigmoid layer.
            gate_q = 1 + torch.sigmoid(self.dyLinear_q(pool_embedding))
            gate_k = 1 + torch.sigmoid(self.dyLinear_k(pool_embedding))

            mixed_query_layer = mixed_query_layer * gate_q.unsqueeze(1)
            mixed_key_layer = mixed_key_layer * gate_k.unsqueeze(1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the
        # raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel
        # forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
        else:
            attn_data = {}

        return context_layer, attn_data


class BertImageSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(
        self,
        input_tensor: Tensor,
        attention_mask: Tensor,
        txt_embedding: Tensor,
        txt_attention_mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        self_output, attention_probs = self.self(
            input_tensor, attention_mask, txt_embedding, txt_attention_mask
        )
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertImageIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_intermediate_size)
        if isinstance(config.v_hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.v_hidden_act]
        else:
            self.intermediate_act_fn = config.v_hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_intermediate_size, config.v_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        txt_embedding: Tensor,
        txt_attention_mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask, txt_embedding, txt_attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs

    @torch.no_grad()
    def forward_no_grad(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        txt_embedding: Tensor,
        txt_attention_mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        return self.forward(
            hidden_states, attention_mask, txt_embedding, txt_attention_mask
        )


class BertBiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.bi_hidden_size, config.bi_num_attention_heads)
            )

        self.visualization = config.visualization
        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(
            config.bi_hidden_size / config.bi_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)

        self.query2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_tensor1: Tensor,
        attention_mask1: Tensor,
        input_tensor2: Tensor,
        attention_mask2: Tensor,
        co_attention_mask: Optional[Tensor] = None,
        use_co_attention_mask: bool = False,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw
        # attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1
        # if use_co_attention_mask:
        # attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.functional.softmax(attention_scores1, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the
        # raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel
        # forward() function)

        # we can comment this line for single flow.
        attention_scores2 = attention_scores2 + attention_mask2
        # if use_co_attention_mask:
        # attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.functional.softmax(attention_scores2, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(new_context_layer_shape2)

        attn_data = {}

        if self.visualization:
            attn_data = {
                "attn1": attention_probs1,
                "queries1": query_layer2,
                "keys1": key_layer1,
                "attn2": attention_probs2,
                "querues2": query_layer1,
                "keys2": key_layer2,
            }

        return context_layer1, context_layer2, attn_data


class BertBiOutput(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.LayerNorm1 = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.q_dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.q_dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.q_dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.q_dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states1: Tensor,
        input_tensor1: Tensor,
        hidden_states2: Tensor,
        input_tensor2: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertConnectionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.biattention = BertBiAttention(config)

        self.biOutput = BertBiOutput(config)

        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)

        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(
        self,
        input_tensor1: Tensor,
        attention_mask1: Tensor,
        input_tensor2: Tensor,
        attention_mask2: Tensor,
        co_attention_mask: Optional[Tensor] = None,
        use_co_attention_mask: bool = False,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:

        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask,
            use_co_attention_mask,
        )

        attention_output1, attention_output2 = self.biOutput(
            bi_output2, input_tensor1, bi_output1, input_tensor2
        )

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.

        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = config.v_biattention_id
        self.t_biattention_id = config.t_biattention_id
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_v_layer = config.fixed_v_layer
        layer = BertLayer(config)
        v_layer = BertImageLayer(config)
        connect_layer = BertConnectionLayer(config)

        self.layer = nn.ModuleList(
            [deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )
        self.v_layer = nn.ModuleList(
            [deepcopy(v_layer) for _ in range(config.v_num_hidden_layers)]
        )
        self.c_layer = nn.ModuleList(
            [deepcopy(connect_layer) for _ in range(len(config.v_biattention_id))]
        )

    def forward(
        self,
        txt_embedding: Tensor,
        image_embedding: Tensor,
        txt_attention_mask: Tensor,
        txt_attention_mask2: Tensor,
        image_attention_mask: Tensor,
        co_attention_mask: Tensor,
        output_all_encoded_layers: bool = True,
        output_all_attention_masks: bool = False,
    ) -> Tuple[
        List[Tensor],
        List[Tensor],
        Tuple[List[Tensor], List[Tensor], List[Tuple[Tensor, Tensor]]],
    ]:

        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t: List[Tensor] = []
        all_encoder_layers_v: List[Tensor] = []

        all_attention_mask_t: List[Tensor] = []
        all_attnetion_mask_v: List[Tensor] = []
        all_attention_mask_c: List[Tuple[Tensor, Tensor]] = []

        batch_size, num_words, t_hidden_size = txt_embedding.size()
        _, num_regions, v_hidden_size = image_embedding.size()

        use_co_attention_mask = False
        for v_layer_id, t_layer_id in zip(self.v_biattention_id, self.t_biattention_id):

            v_end = v_layer_id
            t_end = t_layer_id

            assert self.fixed_t_layer <= t_end
            assert self.fixed_v_layer <= v_end

            cur_idx = 0
            for cur_layer in self.layer:
                if t_start <= cur_idx < self.fixed_t_layer:
                    txt_embedding, txt_attention_probs = cur_layer.forward_no_grad(
                        txt_embedding, txt_attention_mask
                    )
                    t_start = self.fixed_t_layer
                    if output_all_attention_masks and "attn" in txt_attention_probs:
                        all_attention_mask_t.append(txt_attention_probs["attn"])
                cur_idx += 1

            cur_idx = 0
            for cur_layer in self.layer:
                if t_start <= cur_idx < t_end:
                    txt_embedding, txt_attention_probs = cur_layer(
                        txt_embedding, txt_attention_mask
                    )
                    if output_all_attention_masks and "attn" in txt_attention_probs:
                        all_attention_mask_t.append(txt_attention_probs["attn"])
                cur_idx += 1

            cur_v_idx = 0
            for cur_v_layer in self.v_layer:
                if v_start <= cur_v_idx < self.fixed_v_layer:
                    (
                        image_embedding,
                        image_attention_probs,
                    ) = cur_v_layer.forward_no_grad(
                        image_embedding,
                        image_attention_mask,
                        txt_embedding,
                        txt_attention_mask2,
                    )
                    v_start = self.fixed_v_layer

                    if output_all_attention_masks and "attn" in image_attention_probs:
                        all_attnetion_mask_v.append(image_attention_probs["attn"])
                cur_v_idx += 1

            cur_v_idx = 0
            for cur_v_layer in self.v_layer:
                if v_start <= cur_v_idx < v_end:
                    image_embedding, image_attention_probs = cur_v_layer(
                        image_embedding,
                        image_attention_mask,
                        txt_embedding,
                        txt_attention_mask2,
                    )
                    if output_all_attention_masks and "attn" in image_attention_probs:
                        all_attnetion_mask_v.append(image_attention_probs["attn"])
                cur_v_idx += 1

            if count == 0 and self.in_batch_pairs:
                # new batch size is the batch_size ^2
                image_embedding = (
                    image_embedding.unsqueeze(0)
                    .expand(batch_size, batch_size, num_regions, v_hidden_size)
                    .contiguous()
                    .view(batch_size * batch_size, num_regions, v_hidden_size)
                )
                image_attention_mask = (
                    image_attention_mask.unsqueeze(0)
                    .expand(batch_size, batch_size, 1, 1, num_regions)
                    .contiguous()
                    .view(batch_size * batch_size, 1, 1, num_regions)
                )

                txt_embedding = (
                    txt_embedding.unsqueeze(1)
                    .expand(batch_size, batch_size, num_words, t_hidden_size)
                    .contiguous()
                    .view(batch_size * batch_size, num_words, t_hidden_size)
                )
                txt_attention_mask = (
                    txt_attention_mask.unsqueeze(1)
                    .expand(batch_size, batch_size, 1, 1, num_words)
                    .contiguous()
                    .view(batch_size * batch_size, 1, 1, num_words)
                )
                co_attention_mask = (
                    co_attention_mask.unsqueeze(1)
                    .expand(batch_size, batch_size, 1, num_regions, num_words)
                    .contiguous()
                    .view(batch_size * batch_size, 1, num_regions, num_words)
                )

            if count == 0 and self.FAST_MODE:
                txt_embedding = txt_embedding.expand(
                    image_embedding.size(0),
                    txt_embedding.size(1),
                    txt_embedding.size(2),
                )
                txt_attention_mask = txt_attention_mask.expand(
                    image_embedding.size(0),
                    txt_attention_mask.size(1),
                    txt_attention_mask.size(2),
                    txt_attention_mask.size(3),
                )

            if self.with_coattention:
                cur_c_idx = 0
                for cur_c_layer in self.c_layer:
                    if cur_c_idx == count:
                        # do the bi attention.
                        (
                            image_embedding,
                            txt_embedding,
                            co_attention_probs,
                        ) = cur_c_layer(
                            image_embedding,
                            image_attention_mask,
                            txt_embedding,
                            txt_attention_mask,
                            co_attention_mask,
                            use_co_attention_mask,
                        )

                        if (
                            output_all_attention_masks
                            and "attn1" in co_attention_probs
                            and "attn2" in co_attention_probs
                        ):
                            all_attention_mask_c.append(
                                (
                                    co_attention_probs["attn1"],
                                    co_attention_probs["attn2"],
                                )
                            )
                    cur_c_idx += 1

            v_start = v_end
            t_start = t_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)

        cur_v_idx = 0
        for cur_v_layer in self.v_layer:
            if cur_v_idx >= v_start:
                image_embedding, image_attention_probs = cur_v_layer(
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask2,
                )
                if output_all_attention_masks and "attn" in image_attention_probs:
                    all_attnetion_mask_v.append(image_attention_probs["attn"])
            cur_v_idx += 1

        cur_idx = 0
        for cur_layer in self.layer:
            if cur_idx >= t_start:
                txt_embedding, txt_attention_probs = cur_layer(
                    txt_embedding, txt_attention_mask
                )
                if output_all_attention_masks and "attn" in txt_attention_probs:
                    all_attention_mask_t.append(txt_attention_probs["attn"])
            cur_idx += 1

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)

        return (
            all_encoder_layers_t,
            all_encoder_layers_v,
            (all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c),
        )


class BertTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImagePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImgPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.v_hidden_act
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertImagePredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertImgPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.v_hidden_size, config.v_target_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.bi_seq_relationship = nn.Linear(config.bi_hidden_size, 2)
        self.imagePredictions = BertImagePredictionHead(config)
        self.fusion_method = config.fusion_method
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        sequence_output_t: Tensor,
        sequence_output_v: Tensor,
        pooled_output_t: Tensor,
        pooled_output_v: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise AssertionError

        prediction_scores_t = self.predictions(sequence_output_t)
        seq_relationship_score = self.bi_seq_relationship(pooled_output)
        prediction_scores_v = self.imagePredictions(sequence_output_v)

        return prediction_scores_t, prediction_scores_v, seq_relationship_score


class BertImageFeatureEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and
    token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)
        self.LayerNorm = nn.LayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, image_feature: Tensor, image_location: Tensor) -> Tensor:

        img_embeddings = self.image_embeddings(image_feature)
        loc_embeddings = self.image_location_embeddings(image_location)

        # TODO: we want to make the padding_idx==0, however, with custom initilization,
        # it seems it will have a bias. Let's do masking for now
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ViLBERTBase(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Replace transformer layers with scriptable JIT layers
        replace_with_jit()

        # initilize word embedding
        self.embeddings = BertEmbeddings(config)

        self.task_specific_tokens = config.task_specific_tokens

        # initlize the vision embedding
        self.v_embeddings = BertImageFeatureEmbeddings(config)

        self.encoder = BertEncoder(config)
        self.t_pooler = BertTextPooler(config)
        self.v_pooler = BertImagePooler(config)

        self.init_weights()

    def forward(
        self,
        input_txt: Tensor,
        image_feature: Tensor,
        image_location: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_attention_mask: Optional[Tensor] = None,
        co_attention_mask: Optional[Tensor] = None,
        task_ids: Optional[Tensor] = None,
        output_all_encoded_layers: bool = False,
        output_all_attention_masks: bool = False,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Optional[Tuple[List[Tensor], List[Tensor], List[Tuple[Tensor, Tensor]]]],
        Optional[List[Tensor]],
        Optional[List[Tensor]],
    ]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                image_feature.size(0), image_feature.size(1)
            ).type_as(input_txt)

        all_attention_mask_output: Optional[
            Tuple[List[Tensor], List[Tensor], List[Tuple[Tensor, Tensor]]]
        ] = None
        encoded_layers_t_output: Optional[List[Tensor]] = None
        encoded_layers_v_output: Optional[List[Tensor]] = None
        if self.task_specific_tokens:
            # extend the mask
            mask_tokens = torch.ones(input_txt.size(0), 1, device=input_txt.device)
            attention_mask = torch.cat([mask_tokens, attention_mask], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask2 = attention_mask.unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        if not torch.jit.is_scripting():
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if not torch.jit.is_scripting():
            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            extended_image_attention_mask = extended_image_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        if co_attention_mask is None:
            co_attention_mask = torch.zeros(
                input_txt.size(0), image_feature.size(1), input_txt.size(1)
            ).type_as(extended_image_attention_mask)

        extended_co_attention_mask = co_attention_mask.unsqueeze(1)

        # extended_co_attention_mask = co_attention_mask.unsqueeze(-1)
        extended_co_attention_mask = extended_co_attention_mask * 5.0
        if not torch.jit.is_scripting():
            extended_co_attention_mask = extended_co_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )
        embedding_output = self.embeddings(input_txt, token_type_ids, task_ids)
        v_embedding_output = self.v_embeddings(image_feature, image_location)
        encoded_layers_t, encoded_layers_v, all_attention_mask = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_attention_mask2,
            extended_image_attention_mask,
            extended_co_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if output_all_attention_masks:
            all_attention_mask_output = all_attention_mask
        if output_all_encoded_layers:
            encoded_layers_t_output = encoded_layers_t
            encoded_layers_v_output = encoded_layers_v

        return (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            all_attention_mask_output,
            encoded_layers_t_output,
            encoded_layers_v_output,
        )


class ViLBERTForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = ViLBERTBase.from_pretrained(
            self.config.bert_model_name,
            config=BertConfig.from_dict(
                OmegaConf.to_container(self.config, resolve=True)
            ),
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
        )
        self.cls = BertPreTrainingHeads(config)
        self.vocab_size = self.config.vocab_size
        self.visual_target = config.visual_target
        self.num_negative = config.num_negative
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)

        if self.visual_target == 0:
            self.vis_criterion = nn.KLDivLoss(reduction="none")
        elif self.visual_target == 1:
            self.vis_criterion = nn.MSELoss(reduction="none")
        elif self.visual_target == 2:
            self.vis_criterion = CrossEntropyLoss()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.config.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)

            self.tie_weights()

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning
        them instead.
        """
        self._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def forward(
        self,
        input_ids: Tensor,
        image_feature: Tensor,
        image_location: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        image_attention_mask: Tensor,
        masked_lm_labels: Optional[Tensor] = None,
        image_label: Optional[Tensor] = None,
        image_target: Optional[Tensor] = None,
        output_all_attention_masks: bool = False,
    ) -> Dict[str, Tensor]:
        masked_img_loss: Optional[Tensor] = None
        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            attention_weights,
            _encoded_layers_t_output,
            _encoded_layers_v_output,
        ) = self.bert(
            input_ids,
            image_feature,
            image_location,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks,
        )

        prediction_scores_t, prediction_scores_v, seq_relationship_score = self.cls(
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
        )
        output = {}

        if not torch.jit.is_scripting() and output_all_attention_masks:
            output["attention_weights"] = attention_weights

        if image_label is not None and image_target is not None:
            if self.visual_target == 1:
                img_loss = self.vis_criterion(prediction_scores_v, image_target)
                masked_img_loss = torch.sum(
                    img_loss * torch.eq(image_label, 1).unsqueeze(2).float()
                ) / max(
                    torch.sum(
                        torch.eq(image_label, 1).unsqueeze(2).expand_as(img_loss)
                    ),
                    1,
                )
            elif self.visual_target == 0:
                img_loss = self.vis_criterion(
                    F.log_softmax(prediction_scores_v, dim=2), image_target
                )

                masked_img_loss = torch.sum(
                    img_loss * torch.eq(image_label, 1).unsqueeze(2).float()
                ) / max(torch.sum(torch.eq(image_label, 1)), 0)
            elif self.visual_target == 2:
                # generate negative sampled index.
                num_across_batch = int(self.num_negative * 0.7)
                num_inside_batch = int(self.num_negative * 0.3)

                batch_size, num_regions, _ = prediction_scores_v.size()
                assert batch_size != 0
                # random negative across batches.
                row_across_index = torch.ones(
                    batch_size,
                    num_regions,
                    num_across_batch,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ).random_(0, batch_size - 1)
                col_across_index = torch.ones(
                    batch_size,
                    num_regions,
                    num_across_batch,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ).random_(0, num_regions)

                for i in range(batch_size - 1):
                    row_across_index[i][row_across_index[i] == i] = batch_size - 1
                final_across_index = row_across_index * num_regions + col_across_index

                # random negative inside batches.
                row_inside_index = torch.zeros(
                    batch_size,
                    num_regions,
                    num_inside_batch,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                col_inside_index = torch.ones(
                    batch_size,
                    num_regions,
                    num_inside_batch,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ).random_(0, num_regions - 1)

                for i in range(batch_size):
                    row_inside_index[i] = i
                for i in range(num_regions - 1):
                    col_inside_index[:, i, :][col_inside_index[:, i, :] == i] = (
                        num_regions - 1
                    )
                final_inside_index = row_inside_index * num_regions + col_inside_index

                final_index = torch.cat((final_across_index, final_inside_index), dim=2)

                # Let's first sample where we need to compute.
                predict_v = prediction_scores_v[image_label == 1]
                neg_index_v = final_index[image_label == 1]

                flat_image_target = image_target.view(batch_size * num_regions, -1)
                # we also need to append the target feature at the beginning.
                negative_v = flat_image_target[neg_index_v]
                positive_v = image_target[image_label == 1]
                sample_v = torch.cat((positive_v.unsqueeze(1), negative_v), dim=1)

                # calculate the loss.
                score = torch.bmm(sample_v, predict_v.unsqueeze(2)).squeeze(2)
                masked_img_loss = self.vis_criterion(
                    score,
                    torch.zeros(
                        score.size(0), dtype=input_ids.dtype, device=input_ids.device
                    ),
                )
            if masked_img_loss is not None:
                output["masked_img_loss"] = masked_img_loss.unsqueeze(0)

        if masked_lm_labels is not None:
            masked_lm_loss = self.loss_fct(
                prediction_scores_t.view(-1, self.vocab_size), masked_lm_labels.view(-1)
            )
            output["masked_lm_loss"] = masked_lm_loss.unsqueeze(0)
        # next_sentence_loss = self.loss_fct(
        #     seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
        # )
        # output["next_sentence_loss"] = next_sentence_loss.unsqueeze(0)
        return output


class ViLBERTForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = ViLBERTBase.from_pretrained(
            self.config.bert_model_name,
            config=BertConfig.from_dict(
                OmegaConf.to_container(self.config, resolve=True)
            ),
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
        )

        self.training_head_type = self.config.training_head_type
        self.num_labels = self.config.num_labels
        self.fusion_method = config.fusion_method
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # Create a copy of config since struct mode won't allow direct overrides
        # classifier_config is only needed for initializing the classifier
        classifier_config = deepcopy(config)
        classifier_config.hidden_size = config.bi_hidden_size
        if self.config.training_head_type == "nlvr2":
            classifier_config.hidden_size *= 2
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(classifier_config),
            nn.Linear(classifier_config.hidden_size, self.num_labels),
        )
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.config.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()

            # Classifier needs to be initialized always as it is task specific
            self.classifier.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids: Tensor,
        image_feature: Tensor,
        image_location: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_attention_mask: Optional[Tensor] = None,
        masked_lm_labels: Optional[Tensor] = None,
        image_label: Optional[Tensor] = None,
        image_target: Optional[Tensor] = None,
        next_sentence_label: Optional[Tensor] = None,
        output_all_attention_masks: bool = False,
    ) -> Dict[str, Tensor]:

        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            attention_weights,
            _encoded_layers_t_output,
            _encoded_layers_v_output,
        ) = self.bert(
            input_ids,
            image_feature,
            image_location,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks,
        )

        output = {}

        if not torch.jit.is_scripting() and output_all_attention_masks:
            output["attention_weights"] = attention_weights

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise AssertionError

        if self.training_head_type == "nlvr2":
            pooled_output = pooled_output.view(-1, pooled_output.size(1) * 2)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output["scores"] = reshaped_logits

        return output


@registry.register_model("vilbert")
class ViLBERT(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/vilbert/pretrain.yaml"

    # Backward compatibility
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("bert.bert", "model.bert")
            .replace("bert.cls", "model.cls")
            .replace("bert.classifier", "model.classifier")
        )

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = ViLBERTForPretraining(self.config)
        else:
            self.model = ViLBERTForClassification(self.config)

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def get_image_and_text_features(self, sample_list):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids

        if sample_list.dataset_name == "nlvr2":
            bert_input_ids = torch.cat([bert_input_ids, bert_input_ids])
            bert_input_mask = torch.cat([bert_input_mask, bert_input_mask])
            bert_input_type_ids = torch.cat([bert_input_type_ids, bert_input_type_ids])

            # image input
            img0 = getattr(sample_list, "img0", {})
            image_info = getattr(img0, "image_info_0", {})
            image_dim_variable_0 = getattr(image_info, "max_features", None)
            image_feature_variable_0 = getattr(img0, "image_feature_0", None)
            image_location_variable_0 = getattr(image_info, "bbox", None)

            img1 = getattr(sample_list, "img1", {})
            image_info = getattr(img1, "image_info_0", {})
            image_dim_variable_1 = getattr(image_info, "max_features", None)
            image_feature_variable_1 = getattr(img1, "image_feature_0", None)
            image_location_variable_1 = getattr(image_info, "bbox", None)

            image_feature_variable = torch.cat(
                [image_feature_variable_0, image_feature_variable_1]
            )
            image_location_variable = torch.cat(
                [image_location_variable_0, image_location_variable_1]
            )
            image_dim_variable = torch.cat([image_dim_variable_0, image_dim_variable_1])
            image_label_variable = None
            image_target_variable = None
        else:
            image_info = getattr(sample_list, "image_info_0", {})
            image_dim_variable = getattr(image_info, "max_features", None)
            image_feature_variable = getattr(sample_list, "image_feature_0", None)
            image_label_variable = getattr(sample_list, "image_labels", None)
            image_location_variable = getattr(image_info, "bbox", None)

            cls_prob = getattr(image_info, "cls_prob", None)
            image_target = np.array(cls_prob, dtype=np.float32)
            image_target_variable = torch.tensor(
                image_target, dtype=torch.float, device=bert_input_ids.device
            )

        return {
            "input_ids": bert_input_ids,
            "attention_mask": bert_input_mask,
            "token_type_ids": bert_input_type_ids,
            "image_dim": image_dim_variable,
            "image_feature": image_feature_variable,
            "image_location": image_location_variable,
            "image_target": image_target_variable,
            "image_label": image_label_variable,
        }

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def forward(self, sample_list):
        params = self.get_image_and_text_features(sample_list)
        # pretraining labels
        params["masked_lm_labels"] = getattr(sample_list, "lm_label_ids", None)
        # is_random_next = getattr(sample_list, "is_correct", None)
        # TODO(aps): Fix on dataset side
        # params["is_random_next"] = None

        # Prepare Mask
        if params["image_feature"] is not None and params["image_dim"] is not None:
            image_mask = torch.arange(
                params["image_feature"].size(-2), device=params["image_feature"].device
            ).expand(*params["image_feature"].size()[:-1])
            if len(params["image_dim"].size()) < len(image_mask.size()):
                params["image_dim"] = params["image_dim"].unsqueeze(-1)
                assert len(params["image_dim"].size()) == len(image_mask.size())
            image_mask = image_mask < params["image_dim"]
            params["image_attention_mask"] = image_mask.long()
        else:
            params["image_attention_mask"] = None
        params.pop("image_dim")

        output_dict = self.model(
            params["input_ids"],
            params["image_feature"],
            params["image_location"],
            params["token_type_ids"],
            params["attention_mask"],
            params["image_attention_mask"],
            params["masked_lm_labels"],
            params["image_label"],
            params["image_target"],
        )

        if self.config.training_head_type == "pretraining":
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            output_dict["losses"] = {}
            output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
                "masked_lm_loss"
            )
            output_dict["losses"][loss_key + "/masked_img_loss"] = output_dict.pop(
                "masked_img_loss"
            )
            # if params["is_random_next"] is not None:
            #     output_dict["losses"][loss_key + "/next_sentence_loss"]
            #       = output_dict.pop("next_sentence_loss")

        return output_dict
