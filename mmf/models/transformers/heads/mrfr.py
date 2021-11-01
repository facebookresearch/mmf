# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/ChenRocks/UNITER/
# and adapted for MMF.

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead
from mmf.models.transformers.heads.utils import compute_masked_hidden
from torch import Tensor, nn


@registry.register_transformer_head("mrfr")
class MRFR(BaseTransformerHead):
    """
    Masked Region Feature Regression transformer head,
    From uniter paper https://arxiv.org/pdf/1909.11740.pdf
    For an example usage take a look at the unit test.
    """

    @dataclass
    class Config(BaseTransformerHead.Config):
        type: str = "mrfr"
        hidden_size: int = 768
        loss_name: str = "mrfr_loss"
        mrfr_target_key: str = "mrfr_region_target"
        mrfr_mask_key: str = "mrfr_region_mask"
        img_dim: int = 2048
        eps: float = 1e-12

    def __init__(
        self, config: Config, img_embedding_weight: nn.Parameter, *args, **kwargs
    ):
        super().__init__(config, *args, **kwargs)

        # Head modules
        hidden_size = self.config.hidden_size
        assert img_embedding_weight is not None and tuple(
            img_embedding_weight.shape
        ) == (hidden_size, self.config.img_dim), (
            "MRFR head requires 'img_embedding_weight' with shape "
            + f"({hidden_size}, {self.config.img_dim})."
        )

        self.linear_proj_weight = img_embedding_weight
        self.linear_proj_bias = nn.Parameter(torch.zeros(self.config.img_dim))

        self.feat_regress = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=self.config.eps),
        )

    def forward(
        self,
        sequence_output: Tensor,
        processed_sample_list: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        assert (
            processed_sample_list is not None
        ), "MRFR head requires 'processed_sample_list' argument"

        output_dict = {}

        assert (
            self.config.mrfr_target_key in processed_sample_list
            and processed_sample_list[self.config.mrfr_target_key] is not None
        ), (
            f"MRFR pretraining requires {self.config.mrfr_target_key} to be in sample "
            + "list with value not None."
        )
        # (bs*num_feat, img_dim)  Look at unit test for example usage!
        feat_targets = processed_sample_list[self.config.mrfr_target_key]

        assert (
            self.config.mrfr_mask_key in processed_sample_list
            and processed_sample_list[self.config.mrfr_mask_key] is not None
        ), (
            f"MRFR pretraining requires {self.config.mrfr_mask_key} to be in sample "
            + "list with value not None."
        )
        # (bs, num_feat)
        image_region_masks = processed_sample_list[self.config.mrfr_mask_key]

        masked_output = compute_masked_hidden(sequence_output, image_region_masks)
        hidden_states = self.feat_regress(masked_output)
        prediction_feat = F.linear(
            hidden_states, self.linear_proj_weight.t(), self.linear_proj_bias
        )
        mrfr_loss = F.mse_loss(prediction_feat, feat_targets, reduction="mean")

        output_dict["losses"] = {}
        output_dict["losses"][self.config.loss_name] = mrfr_loss
        return output_dict
