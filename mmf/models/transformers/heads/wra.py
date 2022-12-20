# Copyright (c) Facebook, Inc. and its affiliates.

# Initial version was taken from https://github.com/ChenRocks/UNITER/
# and adapted for MMF.

from typing import Dict

from mmf.common.registry import registry
from mmf.modules.ot import optimal_transport_dist
from torch import nn, Tensor


@registry.register_transformer_head("wra")
class WRA(nn.Module):
    """
    Word Region Alignment from UNITER.
    Optimal Transport (OT) distance between text and image
    features is used to optimize for WRA.
    OT transport plan (T) is approximated through IPOT.
    """

    def __init__(
        self,
        loss_name: str = "wra_loss",
        ot_inputs_key: str = "wra_info",
        wra_label_key: str = "is_correct",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.loss_name = loss_name
        self.ot_inputs_key = ot_inputs_key
        self.wra_label_key = wra_label_key

    def forward(
        self,
        sequence_output: Tensor,
        processed_sample_list: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, Dict[str, Tensor]]:
        output_dict = {}

        assert (
            self.ot_inputs_key in processed_sample_list
            and processed_sample_list[self.ot_inputs_key] is not None
        ), (
            f"WRA pretraining requires {self.ot_inputs_key} to be in sample "
            + "list with value not None."
        )
        ot_inputs = processed_sample_list[self.ot_inputs_key]

        assert (
            ot_inputs.get("txt_pad") is not None
            and ot_inputs.get("img_pad") is not None
        ), (
            "WRA pretraining requires 'txt_pad', and 'img_pad' to be in "
            + f"'processed_sample_list[{self.ot_inputs_key}]' with"
            + " values not None."
        )
        assert processed_sample_list.get(self.wra_label_key) is not None, (
            f"WRA pretraining requires {self.wra_label_key} to be in sample "
            + "list with value not None."
        )

        ctx_emb = sequence_output
        tl = processed_sample_list["input_ids"].size(1)
        il = processed_sample_list["image_feat"].size(1)
        txt_emb = ctx_emb[:, :tl, :]
        img_emb = ctx_emb[:, tl : tl + il, :]

        txt_pad = ot_inputs["txt_pad"].bool()
        img_pad = ot_inputs["img_pad"].bool()
        itm_labels = processed_sample_list[self.wra_label_key]
        # NOTE: run in fp32 for stability
        ot_dist = optimal_transport_dist(
            txt_emb.float(), img_emb.float(), txt_pad, img_pad
        ).to(txt_emb)
        ot_pos = ot_dist.masked_select(itm_labels == 1)
        ot_neg = ot_dist.masked_select(itm_labels == 0)
        ot_loss = (ot_pos.sum() - ot_neg.sum()) / (ot_pos.size(0) + ot_neg.size(0))

        output_dict["losses"] = {}
        output_dict["losses"][self.loss_name] = ot_loss
        return output_dict
