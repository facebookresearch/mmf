# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/ChenRocks/UNITER/
# and adapted for MMF.

from typing import Dict

import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.models.transformers.heads.utils import compute_masked_hidden
from torch import nn, Tensor


@registry.register_transformer_head("mrc")
class MRC(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        loss_name: str = "mrc_loss",
        ignore_index: int = -1,
        mrc_label_key: str = "region_class",
        mrc_mask_key: str = "image_region_mask",
        label_dim: int = 1601,
        eps: float = 1e-12,
        use_kl: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.loss_name = loss_name
        self.ignore_index = ignore_index
        self.mrc_label_key = mrc_label_key
        self.mrc_mask_key = mrc_mask_key
        self.use_kl = use_kl

        # Head modules
        self.region_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.Linear(hidden_size, label_dim),
        )

    def forward(
        self,
        sequence_output: Tensor,
        processed_sample_list: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, Dict[str, Tensor]]:
        output_dict = {}
        assert (
            self.mrc_label_key in processed_sample_list
            and processed_sample_list[self.mrc_label_key] is not None
        ), (
            f"MRC pretraining requires {self.mrc_label_key} to be in sample "
            + "list with value not None."
        )
        # (bs*num_feat, label_dim)  Look at unit test for example usage!
        region_labels = processed_sample_list[self.mrc_label_key]

        assert (
            self.mrc_mask_key in processed_sample_list
            and processed_sample_list[self.mrc_mask_key] is not None
        ), (
            f"MRC pretraining requires {self.mrc_mask_key} to be in sample "
            + "list with value not None."
        )
        # (bs, num_feat)
        image_region_masks = processed_sample_list[self.mrc_mask_key]

        masked_output = compute_masked_hidden(sequence_output, image_region_masks)
        prediction_soft_label = self.region_classifier(masked_output)
        if self.use_kl:
            prediction_soft_label = F.log_softmax(prediction_soft_label, dim=-1)
            mrc_loss = F.kl_div(
                prediction_soft_label, region_labels, reduction="batchmean"
            )
        else:
            # background class should not be the target
            label_targets = torch.max(region_labels[:, 1:], dim=-1)[1] + 1
            mrc_loss = F.cross_entropy(
                prediction_soft_label,
                label_targets,
                ignore_index=self.ignore_index,
                reduction="mean",
            )

        output_dict["losses"] = {}
        output_dict["losses"][self.loss_name] = mrc_loss
        return output_dict
