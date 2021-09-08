# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from mmf.common.registry import registry
from mmf.models.transformers.heads.mlp import MLP
from omegaconf import open_dict
from torch import nn
from transformers.modeling_bert import BertPredictionHeadTransform


logger = logging.getLogger()


class PredictionHeadTransformWithInDim(BertPredictionHeadTransform):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.in_dim, config.hidden_size)


@registry.register_transformer_head("multilayer_mlp")
class MultiLayerMLP(MLP):
    @dataclass
    class Config(MLP.Config):
        type: str = "multilayer_mlp"
        in_dim: int = 756
        hidden_size: int = 1536
        num_layers: int = 1

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        with open_dict(config):
            hidden_size = config.hidden_size
            config.hidden_size = config.in_dim
            self.pooler = self.get_pooler(config.pooler_name)(config)
            config.hidden_size = hidden_size

        num_layers = self.config.get("num_layers", 1)
        in_dim = self.config.in_dim
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
        self.in_dim = in_dim

    def forward(
        self,
        sequence_output: torch.Tensor,
        encoded_layers: Optional[List[torch.Tensor]] = None,
        processed_sample_list: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        assert (
            sequence_output.size()[-1] == self.in_dim
        ), "Mismatch between Multilayer MLP head in_dim and sequence_output last dim."
        output_dict = {}
        pooled_output = self.pooler(sequence_output)
        prediction = self.classifier(pooled_output)
        output_dict["scores"] = prediction.view(-1, self.num_labels)
        return output_dict
