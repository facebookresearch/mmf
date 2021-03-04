# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead
from transformers.modeling_bert import BertOnlyMLMHead


LABEL_KEY = "mlm_labels"
COMBINED_LABEL_KEY = "combined_labels"


@registry.register_transformer_head("mlm")
class MLM(BaseTransformerHead):
    @dataclass
    class Config(BaseTransformerHead.Config):
        type: str = "mlm"
        vocab_size: int = 30522
        hidden_size: int = 768
        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-5
        hidden_act: str = "gelu"
        ignore_index: int = -1
        loss_name: str = "masked_lm_loss"

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Head modules
        self.cls = BertOnlyMLMHead(self.config)
        self.vocab_size = self.config.vocab_size

        # Loss
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)

    def tie_weights(self, module: Optional[torch.nn.Module] = None):
        self.cls.predictions.decoder.weight = module.weight

    def forward(
        self,
        sequence_output: torch.Tensor,
        encoded_layers: List[torch.Tensor],
        processed_sample_list: Dict[str, Dict[str, torch.Tensor]],
    ):

        assert (
            LABEL_KEY in processed_sample_list
        ), f"MLM pretraining requires {LABEL_KEY} to be in sample list."

        assert (
            COMBINED_LABEL_KEY in processed_sample_list[LABEL_KEY]
        ), f"labels for all modalities must be concatenated in {COMBINED_LABEL_KEY}"

        output_dict = {}

        masked_labels = processed_sample_list[LABEL_KEY][COMBINED_LABEL_KEY]

        masked_tokens = masked_labels.ne(self.config.ignore_index)
        masked_tokens = torch.where(
            masked_tokens.any(), masked_tokens, masked_tokens.new([True])
        )

        masked_labels = masked_labels[masked_tokens]
        sequence_output = sequence_output[masked_tokens, :]

        prediction = self.cls(sequence_output)
        output_dict["logits"] = prediction
        masked_lm_loss = self.ce_loss(
            prediction.contiguous().view(-1, self.vocab_size),
            masked_labels.contiguous().view(-1),
        )
        output_dict["losses"] = {}
        output_dict["losses"][self.config.loss_name] = masked_lm_loss
        return output_dict
