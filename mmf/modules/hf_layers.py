# Copyright (c) Facebook, Inc. and its affiliates.

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertAttention,
    BertEmbeddings,
    BertEncoder,
    BertLayer,
    BertModel,
    BertPooler,
    BertSelfAttention,
    BertSelfOutput,
)
from transformers.modeling_roberta import (
    RobertaAttention,
    RobertaEmbeddings,
    RobertaEncoder,
    RobertaLayer,
    RobertaModel,
    RobertaSelfAttention,
)
from transformers.modeling_utils import PreTrainedModel


def replace_with_jit():
    """
    Monkey patch some transformer functions to replace with scriptable ones.
    """
    BertEmbeddings.forward = BertEmbeddingsJit.forward
    BertEncoder.forward = BertEncoderJit.forward
    BertLayer.forward = BertLayerJit.forward
    BertAttention.forward = BertAttentionJit.forward
    BertSelfAttention.forward = BertSelfAttentionJit.forward
    BertSelfAttention.transpose_for_scores = BertSelfAttentionJit.transpose_for_scores
    BertModel.forward = BertModelJit.forward
    PreTrainedModel.__jit_unused_properties__ = [
        "base_model",
        "dummy_inputs",
        "device",
        "dtype",
    ]
    RobertaEmbeddings.forward = RobertaEmbeddingsJit.forward
    RobertaEncoder.forward = BertEncoderJit.forward
    RobertaLayer.forward = BertLayerJit.forward
    RobertaAttention.forward = BertAttentionJit.forward
    RobertaSelfAttention.forward = BertSelfAttentionJit.forward
    RobertaSelfAttention.transpose_for_scores = (
        BertSelfAttentionJit.transpose_for_scores
    )
    RobertaModel.forward = BertModelJit.forward


class BertEmbeddingsJit(BertEmbeddings):
    """
    Torchscriptable version of `BertEmbeddings` from Huggingface transformers v2.3.0
    https://github.com/huggingface/transformers/blob/v2.3.0/transformers/modeling_bert.py # noqa

    Modifies the `forward` function

    Changes to `forward` function ::
        Typed inputs and modified device to be input_ids.device by default
    """

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


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
        hidden_states: Tensor,
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


class BertLayerJit(BertLayer):
    """
    Torchscriptable version of `BertLayer` from Huggingface transformers v2.3.0
    https://github.com/huggingface/transformers/blob/v2.3.0/transformers/modeling_bert.py # noqa

    Modifies the `forward` function as well as uses scriptable `BertAttentionJit`

    Changes to `forward` function::
        Typed inputs and modifies the output to be a List[Tensor]
    """

    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionJit(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttentionJit(config)

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


class BertEncoderJit(BertEncoder):
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
        super().__init__(config)
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
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if not torch.jit.is_scripting() and output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                None,
                encoder_hidden_states,
                encoder_attention_mask,
            )
            hidden_states = layer_outputs[0]

            if not torch.jit.is_scripting() and output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if not torch.jit.is_scripting() and output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if not torch.jit.is_scripting():
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertModelJit(BertModel):
    """
    Torchscriptable version of `BertModel` from Huggingface transformers v2.3.0
    https://github.com/huggingface/transformers/blob/v2.3.0/transformers/modeling_bert.py # noqa

    Modifies the `forward` function

    Changes to `forward` function ::
        Typings for input, modifications to device, change output type to
        Tuple[Tensor, Tensor, List[Tensor]]
    """

    __jit_unused_properties__ = ["base_model", "dummy_inputs", "device", "dtype"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddingsJit(config)
        self.encoder = BertEncoderJit(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """Forward pass on the Model.
        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in
        `Attention is all you need`_ by Ashish Vaswani, Noam Shazeer, Niki Parmar,
        Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and
        Illia Polosukhin.
        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.
        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = inputs_embeds.device if inputs_embeds is not None else input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to
            # the padding mask
            # - if the model is an encoder, make the mask broadcastable to
            # [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or "
                + f"attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # Python builtin next is currently not supported in Torchscript
        if not torch.jit.is_scripting():
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to
        # [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output, encoder_outputs[1:])
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class RobertaEmbeddingsJit(RobertaEmbeddings):
    """
    Torchscriptable version of `RobertaEmbeddings` from Huggingface transformers v2.3.0
    https://github.com/huggingface/transformers/blob/v2.3.0/transformers/modeling_roberta.py # noqa

    Modifies the `forward` function

    Changes to `forward` function ::
        Typed inputs and modified device to be input_ids.device by default
    """

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = inputs_embeds.device if inputs_embeds is not None else input_ids.device

        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(
                self.padding_idx + 1,
                seq_length + self.padding_idx + 1,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
