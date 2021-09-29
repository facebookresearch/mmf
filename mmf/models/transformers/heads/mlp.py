# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead
from mmf.modules import layers
from omegaconf import open_dict
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
        pooler_name: str = "bert_pooler"
        num_layers: int = 1

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.num_labels = self.config.num_labels
        self.hidden_size = self.config.hidden_size
        self.in_dim = self.config.in_dim = getattr(
            self.config, "in_dim", self.hidden_size
        )

        # Head modules
        # get_pooler expects hidden_size to be input dim size
        with open_dict(self.config):
            hidden_size = self.config.hidden_size
            self.config.hidden_size = self.config.in_dim
            self.pooler = self.get_pooler(self.config.pooler_name)(self.config)
            self.config.hidden_size = hidden_size

        num_layers = config.get("num_layers", 1)
        assert num_layers >= 0

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Dropout(self.config.hidden_dropout_prob))
            layers.append(PredictionHeadTransformWithInDim(self.config))
            with open_dict(self.config):
                self.config.in_dim = self.config.hidden_size

        self.classifier = nn.Sequential(
            *layers, nn.Linear(self.config.in_dim, self.config.num_labels)
        )

    def forward(
        self,
        sequence_output: torch.Tensor,
        encoded_layers: Optional[List[torch.Tensor]] = None,
        processed_sample_list: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        assert (
            sequence_output.size()[-1] == self.in_dim
        ), "Mismatch between MLP head hidden_size and sequence_output last dim."
        output_dict = {}
        pooled_output = self.pooler(sequence_output)
        prediction = self.classifier(pooled_output)
        output_dict["scores"] = prediction.view(-1, self.num_labels)
        return output_dict

    def get_pooler(self, pooler_name):
        if pooler_name == "bert_pooler":
            return BertPooler
        elif hasattr(layers, pooler_name):
            return getattr(layers, pooler_name)
        else:
            return None


class PredictionHeadTransformWithInDim(BertPredictionHeadTransform):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.in_dim, config.hidden_size)
