# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from mmf.common.registry import registry
from mmf.models.transformers.backends.base import BackendEmbeddings
from mmf.models.transformers.base import BaseTransformerBackend
from mmf.modules.hf_layers import BertModelJit, replace_with_jit
from omegaconf import OmegaConf
from torch import Tensor
from transformers import AutoConfig, AutoModel


class HuggingfaceEmbeddings(BackendEmbeddings):
    def build(self, transformer, transformer_config):
        # Build layers for each modality and initialize
        self.build_layers(
            transformer_config["hidden_size"],
            transformer_config["vocab_size"],
            transformer_config["max_position_embeddings"],
            transformer_config["layer_norm_eps"],
            transformer_config["hidden_dropout_prob"],
            transformer_config["pad_token_id"],
        )
        self.init_weights(transformer, transformer_config["type_vocab_size"])


@registry.register_transformer_backend("huggingface")
class HuggingfaceBackend(BaseTransformerBackend):
    """Transformer backend wih Huggingface transformer models"""

    @dataclass
    class Config(BaseTransformerBackend.Config):
        name: str = "huggingface"
        transformer_params: Dict[str, Any] = field(
            default_factory=lambda: {"transformer_base": "bert-base-uncased"}
        )

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Replace transformer layers with scriptable JIT layers
        replace_with_jit()

    def build_transformer_config(self):
        """Build the transformer base model config."""
        self.transformer_config = AutoConfig.from_pretrained(
            self.config.transformer_params.transformer_base,
            **OmegaConf.to_container(self.config.transformer_params)
        )

    def build_transformer_base(self):
        """Build the transformer base model."""
        hf_params = {"config": self.transformer_config}

        # For BERT models, initialize using Jit version
        if self.config.transformer_params.transformer_base.startswith("bert-"):
            self.transformer = BertModelJit.from_pretrained(
                self.config.transformer_params.transformer_base, **hf_params
            )
        else:
            self.transformer = AutoModel.from_pretrained(
                self.config.transformer_params.transformer_base, **hf_params
            )

    def build_embeddings(self):
        """Build the multimodal embeddings using the transformer base
        embeddings.
        """
        configuration = {
            **self.transformer_config.to_dict(),
            **self.config.transformer_params,
        }
        self.embeddings = HuggingfaceEmbeddings(
            self.modalities, configuration, self.transformer
        )

    def generate_embeddings(
        self,
        tokens_ids: Dict[str, Tensor],
        position_ids: Dict[str, Tensor],
        segment_ids: Dict[str, Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        """Generate multimodal embeddings."""
        return self.embeddings(
            tokens_ids=tokens_ids, position_ids=position_ids, segment_ids=segment_ids
        )

    def generate_attention_mask(self, masks: List[Tensor]) -> Tensor:
        """Generate attention mask."""
        attention_mask = torch.cat(masks, dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def generate_encoded_layers(self, embedding, attention_mask) -> List[Tensor]:
        """Generate the output from transformer layers. For huggingface models the
        output encoded layers is a Tuple(last layer output, all layers). So the
        order is reversed to match the output order of other backends.
        """
        if torch.jit.is_scripting():
            encoded_layers = self.transformer.encoder(embedding, attention_mask)
        else:
            encoded_layers = self.transformer.encoder(
                embedding, attention_mask, [None] * len(self.transformer.encoder.layer)
            )
        return encoded_layers[-1], encoded_layers[0]
