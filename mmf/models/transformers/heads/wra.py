# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/ChenRocks/UNITER/
# and adapted for MMF.

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead
from mmf.modules.ot import optimal_transport_dist


@registry.register_transformer_head("wra")
class WRA(BaseTransformerHead):
    @dataclass
    class Config(BaseTransformerHead.Config):
        type: str = "wra"
        loss_name: str = "wra_loss"
        ot_inputs_key: str = "wra_region_target"
        wra_label_key: str = "is_correct"

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def forward(
        self,
        sequence_output: torch.Tensor,
        processed_sample_list: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        assert (
            processed_sample_list is not None
        ), "WRA head requires 'processed_sample_list' argument"

        output_dict = {}

        assert (
            self.config.ot_inputs_key in processed_sample_list
            and processed_sample_list[self.config.ot_inputs_key] is not None
        ), (
            f"WRA pretraining requires {self.config.ot_inputs_key} to be in sample "
            + "list with value not None."
        )
        ot_inputs = processed_sample_list[self.config.ot_inputs_key]

        ot_scatter = ot_inputs["ot_scatter"]

        b = sequence_output.size(0)
        tl = processed_sample_list.input_ids.size(1)
        il = processed_sample_list.img_feat.size(1)
        max_l = max(ot_inputs["scatter_max"] + 1, tl + il)

        ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
        ctx_emb = torch.zeros(
            b,
            max_l,
            self.config.hidden_size,
            dtype=sequence_output.dtype,
            device=sequence_output.device,
        ).scatter_(dim=1, index=ot_scatter, src=sequence_output)
        txt_emb = ctx_emb[:, :tl, :]
        img_emb = ctx_emb[:, tl : tl + il, :]

        txt_pad = ot_inputs["txt_pad"]
        img_pad = ot_inputs["img_pad"]
        itm_labels = processed_sample_list[self.config.wra_label_key]
        # NOTE: run in fp32 for stability
        ot_dist = optimal_transport_dist(
            txt_emb.float(), img_emb.float(), txt_pad, img_pad
        ).to(txt_emb)
        ot_pos_dist = ot_dist.masked_select(itm_labels == 1)
        ot_neg_dist = ot_dist.masked_select(itm_labels == 0)

        output_dict["losses"] = {}
        output_dict["losses"][self.config.loss_name + "/pos"] = ot_pos_dist
        output_dict["losses"][self.config.loss_name + "/neg"] = ot_neg_dist
        return output_dict

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked
