# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
from typing import Any, Dict, List, Type

import torch
from mmf.models.transformers.base import BaseTransformerModalityConfig
from torch import Tensor, nn


class BackendEmbeddings(nn.Module):
    """Embedding class that can take any number of image or text modalities, each can
    have their input id, position id and segment id. We generate embeddings of
    dimension config.hidden_size for each and then first add the three embeddings
    for each modality to have a modality specific embedding. We then concat the
    modality specific embeddings to have a joint embedding.
    """

    def __init__(
        self,
        modalities: List[BaseTransformerModalityConfig],
        transformer_config: Dict[str, Any],
        transformer: Type[nn.Module],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.modalities = modalities
        self.transformer_config = transformer_config
        self.token_embeddings = nn.ModuleList()
        self.pos_embeddings = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.modality_keys: List = []

        self.build(transformer, self.transformer_config)

        assert (
            len(self.token_embeddings)
            == len(self.pos_embeddings)
            == len(self.layer_norms)
            == len(self.dropouts)
            == len(self.modalities)
        )

    def build(self, transformer, transformer_config):
        # Build layers for each modality and initialize
        self.build_layers()
        self.init_weights(transformer)

    def build_layers(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30255,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        pad_token_id: int = 2,
    ):

        for modality in self.modalities:
            self.modality_keys.append(modality.key)
            layer_norm_eps = modality.get("layer_norm_eps", layer_norm_eps)
            position_dim = modality.get("position_dim", max_position_embeddings)
            hidden_dropout_prob = modality.get(
                "hidden_dropout_prob", hidden_dropout_prob
            )
            if modality.type == "text":
                self.token_embeddings.append(
                    nn.Embedding(
                        vocab_size,
                        hidden_size,
                        padding_idx=pad_token_id,
                    )
                )
            else:
                self.token_embeddings.append(
                    nn.Sequential(
                        nn.Linear(modality.embedding_dim, hidden_size),
                        torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps),
                    )
                )

            self.pos_embeddings.append(nn.Embedding(position_dim, hidden_size))
            self.layer_norms.append(torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps))
            self.dropouts.append(nn.Dropout(hidden_dropout_prob))

        self.token_type_embeddings = nn.Embedding(len(self.modalities), hidden_size)

    def init_weights(
        self,
        transformer: Type[nn.Module],
        type_vocab_size: int = 2,
        token_noise_mean: float = 0.0,
        token_noise_std: float = 0.01,
    ):
        for idx, modality in enumerate(self.modalities):
            if modality.type == "text":
                self.token_embeddings[idx] = transformer.embeddings.word_embeddings
                self.layer_norms[idx] = transformer.embeddings.LayerNorm

            self.pos_embeddings[idx].weight = nn.Parameter(
                deepcopy(transformer.embeddings.position_embeddings.weight.data),
                requires_grad=True,
            )

        # Token Type or Segment Embeddings
        if hasattr(transformer.embeddings, "token_type_embeddings"):
            token_vocab_size = type_vocab_size
            self.token_type_embeddings.weight.data[:token_vocab_size].copy_(
                transformer.embeddings.token_type_embeddings.weight
            )
            for idx in range(token_vocab_size, len(self.modalities)):
                self.token_type_embeddings.weight.data[idx].copy_(
                    transformer.embeddings.token_type_embeddings.weight.data.mean(dim=0)
                )
                # Add random normal noise
                self.token_type_embeddings.weight.data[idx] += torch.normal(
                    token_noise_mean,
                    token_noise_std,
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
