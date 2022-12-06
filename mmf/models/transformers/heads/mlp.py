# Copyright (c) Facebook, Inc. and its affiliates.

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead
from mmf.modules import layers
from omegaconf import OmegaConf, open_dict
from torch import nn

try:
    from transformers3.modeling_bert import BertPooler, BertPredictionHeadTransform
except ImportError:
    from transformers.modeling_bert import BertPooler, BertPredictionHeadTransform


@registry.register_transformer_head("multilayer_mlp")
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
        in_dim: Optional[int] = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.num_labels = self.config.num_labels
        self.hidden_size = self.config.hidden_size
        self.in_dim = self.config.in_dim = (
            self.hidden_size if self.config.in_dim is None else self.config.in_dim
        )

        # Head modules
        # get_pooler expects hidden_size to be input dim size
        pooler_config = OmegaConf.create(dict(self.config, hidden_size=self.in_dim))
        pooler_cls = self.get_pooler(self.config.pooler_name)
        self.pooler = pooler_cls(pooler_config)

        num_layers = config.get("num_layers", 1)
        assert num_layers >= 0

        layers = []
        prediction_head_config = copy.deepcopy(self.config)
        for _ in range(num_layers):
            layers.append(nn.Dropout(self.config.hidden_dropout_prob))
            layers.append(PredictionHeadTransformWithInDim(prediction_head_config))
            with open_dict(prediction_head_config):
                prediction_head_config.in_dim = prediction_head_config.hidden_size

        self.classifier = nn.Sequential(
            *layers, nn.Linear(self.hidden_size, self.num_labels)
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
        elif pooler_name == "identity":
            return nn.Identity
        elif hasattr(layers, pooler_name):
            return getattr(layers, pooler_name)
        else:
            raise NotImplementedError(f"{pooler_name} is not implemented.")


class PredictionHeadTransformWithInDim(BertPredictionHeadTransform):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.in_dim, config.hidden_size)
