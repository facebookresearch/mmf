# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead
from mmf.models.transformers.heads.mlp import MLP

LABEL_KEY = "three_way_constrastive_labels"


@registry.register_transformer_head("contrastive_three_way")
class ThreeWayContrastive(BaseTransformerHead):
    """Three way contrastive loss used for VinVL pretraining.
    Described here https://arxiv.org/pdf/2101.00529

    A thin wrapper around MLP for 3 way classification.
    Effectively ITM with 3 labels.
    contrastive 3-way loss has 3 labels,
    0 for a match, 1, 2 for a corrupt caption/image

    """

    @dataclass
    class Config(BaseTransformerHead.Config):
        type: str = "three_way_contrastive"
        hidden_size: int = 768
        loss_name: str = "three_way_contrastive_loss"
        ignore_index: int = -1
        constrastive_label_key: str = "contrastive_labels"
        num_layers: int = 0
        num_labels: int = 3
        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-6
        hidden_act: str = "gelu"
        pooler_name: str = "bert_pooler"
        in_dim: Optional[int] = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # Head modules
        self.contrast_head = MLP(config=self.config)
        # Loss
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)

    def forward(
        self,
        sequence_output: torch.Tensor,
        processed_sample_list: Dict[str, Dict[str, torch.Tensor]],
    ):
        output_dict = {}

        if self.config.constrastive_label_key in processed_sample_list:
            next_sentence_labels = processed_sample_list[
                self.config.constrastive_label_key
            ]
        else:
            assert (
                LABEL_KEY in processed_sample_list
                and processed_sample_list[LABEL_KEY] is not None
            ), (
                f"Constrastive three way pretraining requires {LABEL_KEY} to "
                + "be in sample list with value not None."
            )

            next_sentence_labels = processed_sample_list[LABEL_KEY][
                self.config.constrastive_label_key
            ]

        scores = self.contrast_head(sequence_output)["scores"]
        constrastive_loss = self.ce_loss(
            scores.contiguous().view(-1, 3),
            next_sentence_labels.contiguous().view(-1),
        )
        output_dict["losses"] = {}
        output_dict["losses"][self.config.loss_name] = constrastive_loss
        return output_dict
