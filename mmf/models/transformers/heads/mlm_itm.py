# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from torch import nn


@registry.register_transformer_head("mlm_itm")
class MLMAndITM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.mlm_head = registry.get_transformer_head_class("mlm")(config.mlm_config)
        self.itm_head = registry.get_transformer_head_class("itm")(config.itm_config)
        self.mlm_loss_weight = config.get("mlm_loss_weight", 1.0)
        self.itm_loss_weight = config.get("itm_loss_weight", 1.0)

    def forward(self, sequence_output, processed_sample_list):
        mlm_outputs = self.mlm_head(
            sequence_output, processed_sample_list=processed_sample_list
        )
        itm_outputs = self.itm_head(
            sequence_output, processed_sample_list=processed_sample_list
        )

        outputs = mlm_outputs
        outputs["losses"]["masked_lm_loss"] *= self.mlm_loss_weight
        outputs["losses"]["itm_loss"] = itm_outputs["losses"]["itm_loss"]
        outputs["losses"]["itm_loss"] *= self.itm_loss_weight

        return outputs
