# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead

try:
    from transformers3.modeling_bert import BertOnlyNSPHead, BertPooler
except ImportError:
    from transformers.modeling_bert import BertOnlyNSPHead, BertPooler


LABEL_KEY = "itm_labels"


@registry.register_transformer_head("itm")
class ITM(BaseTransformerHead):
    @dataclass
    class Config(BaseTransformerHead.Config):
        type: str = "itm"
        hidden_size: int = 768
        loss_name: str = "itm_loss"
        ignore_index: int = -1
        itm_label_key: str = "is_correct"

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Head modules
        self.pooler = BertPooler(self.config)
        self.cls = BertOnlyNSPHead(self.config)

        # Loss
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)

    def forward(
        self,
        sequence_output: torch.Tensor,
        encoded_layers: Optional[List[torch.Tensor]] = None,
        processed_sample_list: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        assert (
            processed_sample_list is not None
        ), "ITM head requires 'processed_sample_list' argument"

        output_dict = {}

        if self.config.itm_label_key in processed_sample_list:
            next_sentence_labels = processed_sample_list[self.config.itm_label_key]
        else:
            assert (
                LABEL_KEY in processed_sample_list
                and processed_sample_list[LABEL_KEY] is not None
            ), (
                f"ITM pretraining requires {LABEL_KEY} to be in sample "
                + "list with value not None."
            )

            next_sentence_labels = processed_sample_list[LABEL_KEY][
                self.config.itm_label_key
            ]

        pooled_output = self.pooler(sequence_output)
        seq_relationship_score = self.cls(pooled_output)
        itm_loss = self.ce_loss(
            seq_relationship_score.contiguous().view(-1, 2),
            next_sentence_labels.contiguous().view(-1),
        )
        output_dict["losses"] = {}
        output_dict["losses"][self.config.loss_name] = itm_loss
        return output_dict
