# Copyright (c) Facebook, Inc. and its affiliates.

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertAttention,
    BertIntermediate,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)


class BertSelfAttentionJit(BertSelfAttention):
    """
    Torchscriptable version of `BertSelfAttention` from Huggingface transformers v2.3.0
    https://github.com/huggingface/transformers/blob/v2.3.0/transformers/modeling_bert.py # noqa

    Modifies the `forward` function and `transpose_for_scores` function

    Changes to `transpose_for_scores` function ::
        Changes the `new_x_shape` unpacking as static size inference is not supported

    Changes to `forward` function ::
        Uses scriptable `nn.functional.softmax` and also removes several static size
        inference which is not supported.
    """

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
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

        # Take the dot product between "query" and "key" to get the raw attention
        # scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel
            # forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs)
        return outputs


class BertAttentionJit(BertAttention):
    """
    Torchscriptable version of `BertAttention` from Huggingface transformers v2.3.0
    https://github.com/huggingface/transformers/blob/v2.3.0/transformers/modeling_bert.py # noqa

    Modifies the `forward` function as well as uses scriptable `BertSelfAttentionJit`

    Changes to `forward` function ::
        Typed inputs and modifies the output to be a List[Tensor]
    """

    def __init__(self, config):
        super().__init__(config)
        self.self = BertSelfAttentionJit(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class BertLayerJit(nn.Module):
    """
    Torchscriptable version of `BertLayer` from Huggingface transformers v2.3.0
    https://github.com/huggingface/transformers/blob/v2.3.0/transformers/modeling_bert.py # noqa

    Modifies the `forward` function as well as uses scriptable `BertAttentionJit`

    Changes to `forward` function::
        Typed inputs and modifies the output to be a List[Tensor]
    """

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttentionJit(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoderJit(nn.Module):
    """
    Torchscriptable version of `BertEncoder` from Huggingface transformers v2.3.0
    https://github.com/huggingface/transformers/blob/v2.3.0/transformers/modeling_bert.py # noqa

    Modifies the `forward` function as well as uses scriptable `BertLayerJit`

    Changes to `forward` function::
        Typed inputs and modifies the output to be of Tuple[Tensor] type in scripting
        mode. Due to different possible types when `output_hidden_states` or
        `output_attentions` are enable, we do not support these in scripting mode
    """

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [BertLayerJit(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if not torch.jit.is_scripting() and self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                None,
                encoder_hidden_states,
                encoder_attention_mask,
            )
            hidden_states = layer_outputs[0]

            if not torch.jit.is_scripting() and self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if not torch.jit.is_scripting() and self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if not torch.jit.is_scripting():
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
