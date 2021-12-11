# Copyright (c) Facebook, Inc. and its affiliates.

# Code based off https://github.com/microsoft/Oscar
# modified for MMF
# Licensed under the MIT license.

import logging
from collections import namedtuple
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
)

logger = logging.getLogger(__name__)


class VinVLBase(BertPreTrainedModel):
    """VinVL Bert Encoder for image features
    From https://github.com/microsoft/Oscar/blob/master/oscar/modeling/modeling_bert.py
    Is a thin wrapper around BertEncoder that handles image features
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.img_dim = config.img_feature_dim
        self.use_img_layernorm = getattr(config, "use_img_layernorm", False)

        img_projection = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        img_embedding_list = [img_projection]
        if self.use_img_layernorm:
            img_embedding_list += [
                nn.LayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)
            ]
        dropout = nn.Dropout(config.hidden_dropout_prob)
        img_embedding_list += [dropout]
        # is an image encoding used as input to the transformer trunk
        self.img_embedding = nn.Sequential(*img_embedding_list)

    def forward(
        self,
        input_ids: Tensor,
        img_feats: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.size(0), input_ids.size(1) + img_feats.size(1))
            ).to(input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # attention_mask with dim 3 is to specify a unique mask for each feature,
        # it is broadcast over heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # Make the mask broadcastable to
            # [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_ids.shape})"
                + " or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Do embeddings
        text_embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        img_embedding_output = self.img_embedding(img_feats)
        embedding_output = torch.cat((text_embedding_output, img_embedding_output), 1)

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_hidden_states=True,
        )
        layers = namedtuple("TransformerOutput", ["last_hidden_state", "hidden_layers"])
        return layers(encoder_outputs[0], encoder_outputs[1])
