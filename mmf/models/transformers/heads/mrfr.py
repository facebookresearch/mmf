# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/ChenRocks/UNITER/
# and adapted for MMF.

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead
from torch import nn


LABEL_KEY = "mrfr_targets"


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

    def __init__(self, config: Config, img_embedding_weight, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Head modules
        hidden_size = self.config.hidden_size
        assert img_embedding_weight is not None and tuple(
            img_embedding_weight.shape
        ) == (self.config.img_dim, hidden_size), (
            "MRFR head requires 'img_embedding_weight' with shape "
            + f"({self.config.img_dim}, {hidden_size})."
        )

        self.linear_proj = nn.Linear(hidden_size, self.config.img_dim)
        self.linear_proj.weight.data = img_embedding_weight
        self.linear_proj.bias.data.fill_(0)

        self.feat_regress = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=self.config.eps),
            self.linear_proj,
        )

    def forward(
        self,
        sequence_output: torch.Tensor,
        processed_sample_list: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
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

        masked_output = self._compute_masked_hidden(sequence_output, image_region_masks)
        prediction_feat = self.feat_regress(masked_output)
        mrfr_loss = F.mse_loss(prediction_feat, feat_targets, reduction="mean")

        output_dict["losses"] = {}
        output_dict["losses"][self.config.loss_name] = mrfr_loss
        return output_dict

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked
