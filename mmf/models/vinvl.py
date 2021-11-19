# Copyright (c) Facebook, Inc. and its affiliates.

# Code based off https://github.com/microsoft/Oscar
# modified for MMF
# Licensed under the MIT license.

import logging
from collections import namedtuple
from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
)

logger = logging.getLogger(__name__)

EMPTY_CONFIG = OmegaConf.create({})
NUM_RETRIES = 6


class BertImgModel(BertPreTrainedModel):
    """VinVL Bert Encoder for image features
    From https://github.com/microsoft/Oscar/blob/master/oscar/modeling/modeling_bert.py
    Is a thin wrapper around BertEncoder that handles image features
    """

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.img_dim = config.img_feature_dim
        self.use_img_layernorm = getattr(config, "use_img_layernorm", False)

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = nn.LayerNorm(
                config.hidden_size, eps=config.img_layer_norm_eps
            )

    def forward(
        self,
        input_ids: Tensor,
        img_feats: Tensor,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple(Tensor):
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.size(0), input_ids.size(1) + img_feats.size(1))
            ).to(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular
        # masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = self._get_head_mask(head_mask)

        # Do embeddings
        text_embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        img_embedding_output = self.img_embedding(img_feats)
        if self.use_img_layernorm:
            img_embedding_output = self.LayerNorm(img_embedding_output)
        img_embedding_output = self.dropout(img_embedding_output)
        embedding_output = torch.cat((text_embedding_output, img_embedding_output), 1)

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_hidden_states=True,
        )
        layers = namedtuple("TransformerOutput", ["final_layer", "hidden_layers"])
        return layers(encoder_outputs[0], encoder_outputs[1])

    def _get_head_mask(self, head_mask):
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        return head_mask
