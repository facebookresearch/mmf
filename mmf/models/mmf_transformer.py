# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, List

import torch
from mmf.common.registry import registry
from mmf.common.typings import DictConfig
from mmf.models.transformers.base import (
    BaseTransformer,
    BaseTransformerConfigType,
    BaseTransformerInput,
)
from mmf.modules.encoders import MultiModalEncoderBase
from torch import Tensor, nn
from transformers.modeling_bert import BertPooler, BertPredictionHeadTransform


class ImageEncoder(MultiModalEncoderBase):
    """Extends the MultiModalEncoderBase class which builds the encoder based on
    the config parameters. We can set the type of image encoder(resnet50, resnet152,
    resnext50 etc) and other parameters like num of features, type of pooling etc.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def build(self):
        self.encoder = self._build_modal_encoder(self.config.image_encoder)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


@registry.register_model("mmf_transformer")
class MMFTransformer(BaseTransformer):
    def __init__(self, config: BaseTransformerConfigType, *args, **kwargs):
        super().__init__(config)
        self.num_labels = self.config.num_labels
        self.modality_keys: List = []
        self.modality_type: List = []
        self.modality_segments: List = []
        for modality in self.config.modalities:
            self.modality_keys.append(modality.key)
            self.modality_type.append(modality.type)
            if "segment_id" in modality:
                self.modality_segments.append(modality.segment_id)
            else:
                self.modality_segments.append(-1)

    @classmethod
    def config_path(cls) -> str:
        return "configs/models/mmf_transformer/defaults.yaml"

    def build_encoders(self):
        self.image_encoder = ImageEncoder(self.config)
        if getattr(self.config, "freeze_image_encoder", False):
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def build_heads(self):
        """Initialize the classifier head. It takes the output of the
        transformer encoder and passes it through a pooler (we use the pooler from BERT
        model), then dropout, BertPredictionHeadTransform (which is a linear layer,
        followed by activation and layer norm) and lastly a linear layer projecting the
        hidden output to classification labels.
        """
        transformer_config = self.backend.get_config()
        self.pooler = BertPooler(transformer_config)
        self.classifier = nn.Sequential(
            nn.Dropout(transformer_config.hidden_dropout_prob),
            BertPredictionHeadTransform(transformer_config),
            nn.Linear(transformer_config.hidden_size, self.config.num_labels),
        )

    def preprocess_sample(self, sample_list: Dict[str, Tensor]) -> BaseTransformerInput:
        """Preprocess the sample list elements and form a BaseTransformerInput
        type object. This object standardizes how we represent multiple modalities.
        Check the definition of this dataclass in BaseTransformer.
        """

        # Input IDs (or text tokens/image features)
        input_ids: Dict[str, Tensor] = {}
        for idx, modality in enumerate(self.modality_keys):
            if self.modality_type[idx] == "text":
                if sample_list["input_ids"].dim() > 2:
                    input_ids[modality] = sample_list["input_ids"][:, idx]
                else:
                    input_ids[modality] = sample_list["input_ids"]
            elif self.modality_type[idx] == "image":
                if "image" in sample_list:
                    image_modal = sample_list["image"]
                else:
                    image_modal = sample_list["image_feature_0"]
                input_ids[modality] = self.image_encoder(image_modal)

        # Position IDs
        position_ids: Dict[str, Tensor] = {}
        for modality in self.modality_keys:
            position_ids[modality] = (
                torch.arange(
                    0,
                    input_ids[modality].size(1),
                    dtype=torch.long,
                    device=input_ids[modality].device,
                )
                .unsqueeze(0)
                .expand(input_ids[modality].size()[:2])
            )

        # Segment IDs
        segment_ids: Dict[str, Tensor] = {}
        for idx, modality in enumerate(self.modality_keys):
            if self.modality_segments[idx] == -1:
                continue
            if self.modality_type[idx] == "text" and "segment_ids" in sample_list:
                if sample_list["segment_ids"].dim() > 2:
                    segment_ids[modality] = sample_list["segment_ids"][:, idx]
                else:
                    segment_ids[modality] = sample_list["segment_ids"]
            else:
                segment_ids[modality] = torch.zeros(
                    input_ids[modality].size()[:2],
                    dtype=torch.long,
                    device=input_ids[modality].device,
                ).fill_(self.modality_segments[idx])

        # Masks
        masks: Dict[str, Tensor] = {}
        for idx, modality in enumerate(self.modality_keys):
            if self.modality_type[idx] == "text":
                if sample_list["input_mask"].dim() > 2:
                    masks[modality] = sample_list["input_mask"][:, idx]
                else:
                    masks[modality] = sample_list["input_mask"]

            elif self.modality_type[idx] == "image":
                if "image_mask" in sample_list:
                    masks[modality] = sample_list["image_mask"]
                else:
                    masks[modality] = torch.ones(
                        input_ids[modality].size()[:-1],
                        dtype=torch.long,
                        device=input_ids[modality].device,
                    )

        return BaseTransformerInput(input_ids, position_ids, segment_ids, masks)

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Sample preprocess
        output = self.preprocess_sample(sample_list)

        # Arrange masks in a list
        masks = []
        for modality in self.modality_keys:
            masks.append(output.masks[modality])

        # Call transformer backend
        sequence_output, _ = self.backend(
            output.input_ids, output.position_ids, output.segment_ids, masks
        )

        # Transformer Heads
        pooled_output = self.pooler(sequence_output)
        head_output = self.classifier(pooled_output)

        # Postprocess outputs
        return self.postprocess_output(head_output)

    def postprocess_output(self, output: Tensor) -> Dict[str, Tensor]:
        """Postprocess the output from the classifier head and reshape it.
        This will be used to calculate losses and metrics in mmf.
        """
        output_dict = {}
        output_dict["scores"] = output.contiguous().view(-1, self.num_labels)
        return output_dict
