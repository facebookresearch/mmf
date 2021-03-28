# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead
from torch import nn
from transformers.modeling_bert import BertPooler, BertPredictionHeadTransform


@registry.register_transformer_head("mlp")
class MLP(BaseTransformerHead):
    @dataclass
    class Config(BaseTransformerHead.Config):
        type: str = "mlp"
        num_labels: int = 2
        hidden_size: int = 768
        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-6
        hidden_act: str = "gelu"

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Head modules
        self.pooler = BertPooler(self.config)
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            BertPredictionHeadTransform(self.config),
            nn.Linear(self.config.hidden_size, self.config.num_labels),
        )
        self.num_labels = self.config.num_labels
        self.hidden_size = self.config.hidden_size

    def forward(
        self,
        sequence_output: torch.Tensor,
        encoded_layers: Optional[List[torch.Tensor]] = None,
        processed_sample_list: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        assert (
            sequence_output.size()[-1] == self.hidden_size
        ), "Mismatch between MLP head hidden_size and sequence_output last dim."
        output_dict = {}
        pooled_output = self.pooler(sequence_output)
        prediction = self.classifier(pooled_output)
        output_dict["scores"] = prediction.view(-1, self.num_labels)
        return output_dict
