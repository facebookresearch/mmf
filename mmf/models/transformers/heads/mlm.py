# Copyright (c) Facebook, Inc. and its affiliates.

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from mmf.common.registry import registry
from mmf.models.transformers.base import BaseTransformerHead

try:
    from transformers3.modeling_bert import BertOnlyMLMHead
except ImportError:
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
        label_key: Optional[str] = None

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
        encoded_layers: Optional[List[torch.Tensor]] = None,
        processed_sample_list: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        assert (
            processed_sample_list is not None
        ), "MLM head requires 'processed_sample_list' argument"

        output_dict = {}

        if self.config.label_key is not None:
            assert self.config.label_key in processed_sample_list, (
                f"Didn't find label key {self.config.label_key} in "
                + "SampleList required by MLM"
            )
            masked_labels = processed_sample_list[self.config.label_key]
        else:
            assert (
                LABEL_KEY in processed_sample_list
                and processed_sample_list[LABEL_KEY] is not None
            ), (
                f"MLM pretraining requires {LABEL_KEY} to be in sample "
                + "list with value not None."
            )

            assert (
                COMBINED_LABEL_KEY in processed_sample_list[LABEL_KEY]
            ), f"labels for all modalities must be concatenated in {COMBINED_LABEL_KEY}"

            masked_labels = processed_sample_list[LABEL_KEY][COMBINED_LABEL_KEY]

        masked_tokens = masked_labels.ne(self.config.ignore_index)

        masked_labels = masked_labels[masked_tokens]
        sequence_output = sequence_output[masked_tokens, :]

        prediction = self.cls(sequence_output)
        output_dict["logits"] = prediction
        masked_lm_loss = self.ce_loss(
            prediction.contiguous().view(-1, self.vocab_size),
            masked_labels.contiguous().view(-1),
        )
        # When masked_labels are all ignore_index then masked_lm_loss is NaN,
        # so we replace NaN with 0.
        if torch.isnan(masked_lm_loss):
            warnings.warn("NaN detected in masked_lm_loss. Replacing it with 0.")
            masked_lm_loss = torch.nan_to_num(masked_lm_loss, nan=0.0)
        output_dict["losses"] = {}
        output_dict["losses"][self.config.loss_name] = masked_lm_loss
        return output_dict


@registry.register_transformer_head("mlm_multi")
class MLMForMultiHeads(BaseTransformerHead):
    def __init__(self, config):
        super().__init__(config)
        self.mlm_head = MLM(config)

    def forward(self, _, processed_sample_list):
        mlm_outputs = self.mlm_head(
            processed_sample_list["hs_masked_for_mlm"],
            processed_sample_list=processed_sample_list,
        )

        return mlm_outputs
