# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead
from mmf.modules.losses import LogitBinaryCrossEntropy, MSLoss

from .mlp import MLP
from .refiner import Refiner


logger = logging.getLogger(__name__)


class MLPWithLoss(BaseTransformerHead):
    class Config(BaseTransformerHead.Config):
        config: MLP.Config
        loss_name: str = "classification_loss"
        loss: str = "cross_entropy"
        max_sample_size: int = 10000

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.loss_name = self.config.loss_name
        if self.config.loss == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.config.loss == "logit_bce":
            self.loss_fn = LogitBinaryCrossEntropy()
        self.init_output_dict = {}
        self.init_output_dict["losses"] = {}
        self.mlp_base = MLP(config)

    def forward(
        self,
        sequence_output: torch.Tensor,
        encoded_layers: Optional[List[torch.Tensor]] = None,
        processed_sample_list: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        output_dict = self.mlp_base(
            sequence_output, encoded_layers, processed_sample_list
        )
        scores = output_dict["scores"]
        score_max = min(len(scores) - 1, self.config.max_sample_size)
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            if "losses" not in output_dict.keys():
                output_dict["losses"] = {}
            output_dict["losses"][self.loss_name] = self.loss_fn(
                scores[:score_max],
                processed_sample_list["target_key"]["targets"][:score_max],
            )
        elif isinstance(self.loss_fn, LogitBinaryCrossEntropy):
            scores_subset = {}
            scores_subset["scores"] = scores[:score_max]
            targets_subset = {}
            targets_subset["targets"] = processed_sample_list["target_key"]["targets"]
            targets_subset["targets"] = targets_subset["targets"][:score_max]
            if "losses" not in output_dict.keys():
                output_dict["losses"] = {}
            output_dict["losses"][self.loss_name] = self.loss_fn(
                targets_subset, scores_subset
            )

        return output_dict


@registry.register_transformer_head("refiner_classifier")
class RefinerClassifier(BaseTransformerHead):
    @dataclass
    class Config(BaseTransformerHead.Config):
        type: str = "refiner_classifier"
        refiner_config: Optional[Refiner.Config] = None
        mlp_loss_config: Optional[MLPWithLoss.Config] = None
        msloss_weight: float = 0.1
        use_msloss: bool = False
        embedding_key: str = "fused_embedding"
        num_labels: int = 2

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.refiner_head = Refiner(self.config.refiner_config)
        self.mlp_loss_head = MLPWithLoss(self.config.mlp_loss_config)
        self.max_sample_size = self.config.mlp_loss_config.max_sample_size
        self.msloss_weight = self.config.msloss_weight
        if self.config.num_labels > 2:
            self.is_multilabel = True
        else:
            self.is_multilabel = False

        if self.config.use_msloss:
            self.msloss = MSLoss(is_multilabel=self.is_multilabel)
        else:
            self.msloss = None
        self.emb_f = self.config.embedding_key

    def forward(
        self,
        sequence_output: torch.Tensor,
        encoded_layers: Optional[List[torch.Tensor]] = None,
        processed_sample_list: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        output_dict_refiner = self.refiner_head(
            sequence_output, encoded_layers, processed_sample_list
        )
        output_dict = self.mlp_loss_head(
            sequence_output, encoded_layers, processed_sample_list
        )
        for key in output_dict_refiner["losses"].keys():
            if key not in output_dict["losses"].keys():
                output_dict["losses"][key] = output_dict_refiner["losses"][key]

        for key in output_dict_refiner.keys():
            if key not in output_dict.keys():
                output_dict[key] = output_dict_refiner[key]

        scores = output_dict["scores"]

        score_max = min(len(scores) - 1, self.max_sample_size)
        if isinstance(self.msloss, MSLoss):
            emb_f = self.emb_f
            targets_list = {}
            targets_list["targets"] = processed_sample_list["target_key"]["targets"][
                :score_max
            ]
            subset_score_list = {}
            subset_score_list["scores"] = output_dict["scores"][:score_max]
            subset_score_list[emb_f] = output_dict[emb_f][:score_max]
            output_dict["losses"]["ms_loss"] = self.msloss_weight * self.msloss(
                targets_list, subset_score_list
            )

        return output_dict
