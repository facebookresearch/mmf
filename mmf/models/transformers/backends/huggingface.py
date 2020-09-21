# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
from typing import Any, Dict, List, Type

import torch
from mmf.common.registry import registry
from mmf.models.transformers.base import (
    BaseTransformerBackend,
    BaseTransformerConfigType,
)
from mmf.modules.hf_layers import replace_with_jit
from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel


class HuggingfaceEmbeddings(nn.Module):
    """Embedding class that can take any number of image or text modalities, each can
    have their input id, position id and segment id. We generate embeddings of
    dimension config.hidden_size for each and then first add the three embeddings
    for each modality to have a modality specific embedding. We then concat the
    modality specific embeddings to have a joint embedding.
    """

    def __init__(
        self,
        model_config: BaseTransformerConfigType,
        transformer_config: Dict[str, Any],
        transformer: Type[nn.Module],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model_config = model_config
        self.transformer_config = transformer_config

        self.token_embeddings = nn.ModuleList()
        self.pos_embeddings = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.modality_keys: List = []

        # Build layers for each modality and initialize
        self.build_layers()
        self.init_weights(transformer)

        assert (
            len(self.token_embeddings)
            == len(self.pos_embeddings)
            == len(self.layer_norms)
            == len(self.dropouts)
            == len(self.model_config.modalities)
        )

    def build_layers(self):

        for modality in self.model_config.modalities:
            self.modality_keys.append(modality.key)
            layer_norm_eps = modality.get(
                "layer_norm_eps", self.transformer_config.layer_norm_eps
            )
            position_dim = modality.get(
                "position_dim", self.transformer_config.max_position_embeddings
            )
            hidden_dropout_prob = modality.get(
                "hidden_dropout_prob", self.transformer_config.hidden_dropout_prob
            )
            if modality.type == "text":
                self.token_embeddings.append(
                    nn.Embedding(
                        self.transformer_config.vocab_size,
                        self.transformer_config.hidden_size,
                        padding_idx=self.transformer_config.pad_token_id,
                    )
                )
            else:
                self.token_embeddings.append(
                    nn.Sequential(
                        nn.Linear(
                            modality.embedding_dim, self.transformer_config.hidden_size
                        ),
                        torch.nn.LayerNorm(
                            self.transformer_config.hidden_size, eps=layer_norm_eps
                        ),
                    )
                )

            self.pos_embeddings.append(
                nn.Embedding(position_dim, self.transformer_config.hidden_size)
            )
            self.layer_norms.append(
                torch.nn.LayerNorm(
                    self.transformer_config.hidden_size, eps=layer_norm_eps
                )
            )
            self.dropouts.append(nn.Dropout(hidden_dropout_prob))

        self.token_type_embeddings = nn.Embedding(
            len(self.model_config.modalities), self.transformer_config.hidden_size
        )

    def init_weights(self, transformer: Type[nn.Module]):
        for idx, modality in enumerate(self.model_config.modalities):
            if modality.type == "text":
                self.token_embeddings[idx] = transformer.embeddings.word_embeddings
                self.layer_norms[idx] = transformer.embeddings.LayerNorm

            self.pos_embeddings[idx].weight = nn.Parameter(
                deepcopy(transformer.embeddings.position_embeddings.weight.data),
                requires_grad=True,
            )

        # Token Type or Segment Embeddings
        if hasattr(transformer.embeddings, "token_type_embeddings"):
            token_vocab_size = self.transformer_config.type_vocab_size
            self.token_type_embeddings.weight.data[:token_vocab_size].copy_(
                transformer.embeddings.token_type_embeddings.weight
            )
            for idx in range(token_vocab_size, len(self.model_config.modalities)):
                self.token_type_embeddings.weight.data[idx].copy_(
                    transformer.embeddings.token_type_embeddings.weight.data.mean(dim=0)
                )
                # Add random normal noise
                self.token_type_embeddings.weight.data[idx] += torch.normal(
                    self.model_config.token_noise_mean,
                    self.model_config.token_noise_std,
                    size=self.token_type_embeddings.weight.data[idx].size(),
                )

    def forward(
        self,
        tokens_ids: Dict[str, Tensor],
        position_ids: Dict[str, Tensor],
        segment_ids: Dict[str, Tensor],
    ) -> Tensor:
        list_embeddings = []
        for idx, (token_emb, pos_emb, layer_norm, dropout) in enumerate(
            zip(
                self.token_embeddings,
                self.pos_embeddings,
                self.layer_norms,
                self.dropouts,
            )
        ):
            modality_name = self.modality_keys[idx]
            total_embedding = token_emb(tokens_ids[modality_name])
            if modality_name in position_ids:
                total_embedding += pos_emb(position_ids[modality_name])

            if modality_name in segment_ids:
                total_embedding += self.token_type_embeddings(
                    segment_ids[modality_name]
                )

            list_embeddings.append(dropout(layer_norm(total_embedding)))

        return torch.cat(list_embeddings, dim=1)


@registry.register_transformer_backend("huggingface")
class HuggingfaceBackend(BaseTransformerBackend):
    """Transformer backend wih Huggingface transformer models
    """

    def __init__(self, config: BaseTransformerConfigType, *args, **kwargs):
        super().__init__(config)

        # Replace transformer layers with scriptable JIT layers
        replace_with_jit()

    def build_transformer_config(self):
        """Build the transformer base model config.
        """
        self.transformer_config = AutoConfig.from_pretrained(
            self.config.transformer_base, **OmegaConf.to_container(self.config)
        )

    def build_transformer_base(self):
        """Build the transformer base model.
        """
        self.transformer = AutoModel.from_pretrained(
            self.config.transformer_base, config=self.transformer_config
        )

    def build_embeddings(self):
        """Build the multimodal embeddings using the transformer base
        embeddings.
        """
        self.embeddings = HuggingfaceEmbeddings(
            self.config, self.transformer_config, self.transformer
        )

    def get_config(self):
        """Return the transformer configuration.
        """
        return self.transformer_config

    def generate_embeddings(
        self,
        tokens_ids: Dict[str, Tensor],
        position_ids: Dict[str, Tensor],
        segment_ids: Dict[str, Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        """Generate multimodal embeddings.
        """
        return self.embeddings(
            tokens_ids=tokens_ids, position_ids=position_ids, segment_ids=segment_ids
        )

    def generate_attention_mask(self, masks: List[Tensor]) -> Tensor:
        """Generate attention mask.
        """
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
