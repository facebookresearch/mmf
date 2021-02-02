# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from mmf.common.registry import registry
from mmf.models.transformers.base import (
    BaseTransformer,
    BaseTransformerBackendConfig,
    BaseTransformerInput,
    BaseTransformerModalityConfig,
)
from mmf.modules.encoders import ResNet152ImageEncoder
from mmf.utils.build import build_encoder
from omegaconf import MISSING, OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import BertPooler, BertPredictionHeadTransform


@dataclass
class MMFTransformerModalityConfig(BaseTransformerModalityConfig):
    pass


@dataclass
class MMFTransformerBackendConfig(BaseTransformerBackendConfig):
    pass


class MMFTransformerInput(BaseTransformerInput):
    pass


# Can be used with mmft or mmf_transformer
@registry.register_model("mmft")
@registry.register_model("mmf_transformer")
class MMFTransformer(BaseTransformer):
    @dataclass
    class Config(BaseTransformer.Config):
        model: str = "mmft"
        transformer_base: str = "bert-base-uncased"
        training_head_type: str = "classification"
        num_labels: int = MISSING
        initializer_range: float = 0.02
        initializer_mean: float = 0.0
        token_noise_std: float = 0.01
        token_noise_mean: float = 0.0
        layer_norm_weight_fill: float = 1.0
        random_initialize: bool = False
        freeze_transformer: bool = False
        freeze_image_encoder: bool = False
        finetune_lr_multiplier: float = 1
        backend: BaseTransformerBackendConfig = MMFTransformerBackendConfig(
            type="huggingface"
        )
        modalities: List[BaseTransformerModalityConfig] = field(
            default_factory=lambda: [
                MMFTransformerModalityConfig(
                    type="text",
                    key="text",
                    position_dim=512,
                    embedding_dim=768,
                    segment_id=0,
                ),
                MMFTransformerModalityConfig(
                    type="image",
                    key="image",
                    embedding_dim=2048,
                    position_dim=1,
                    segment_id=1,
                    # NOTE: One can also specify encoder in factory mode as
                    # encoder=ImageEncoderFactory.Config(
                    #   type="resnet152",
                    #   params=ResNet152ImageEncoder.Config()
                    # )
                    encoder=ResNet152ImageEncoder.Config(),
                ),
            ]
        )

    def __init__(self, config: BaseTransformer.Config, *args, **kwargs):
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
        self.encoders = nn.ModuleDict()

        for modality in self.config.modalities:
            if "encoder" not in modality:
                # Support "image_encoder" attribute in config if directly provided
                if modality.type == "image" and "image_encoder" in self.config:
                    encoder_config = self.config.image_encoder
                else:
                    # 100 is a random number added to satisfy identity encoder
                    # Set encoder to identity
                    encoder_config = OmegaConf.create(
                        {"type": "identity", "params": {"in_dim": 100}}
                    )
            else:
                encoder_config = modality.encoder

            encoder = build_encoder(encoder_config)
            self.encoders[modality.key] = encoder

            if modality.type == "image" and getattr(
                self.config, "freeze_image_encoder", False
            ):
                for param in encoder.parameters():
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
        for idx, encoder in enumerate(self.encoders.values()):
            modality = self.modality_keys[idx]
            if self.modality_type[idx] == "text":
                # First, check if standard input_ids corresponds to text
                # if not, check for modality key inside the sample list
                if "input_ids" in sample_list:
                    text_ids = sample_list["input_ids"]
                elif modality in sample_list:
                    text_ids = sample_list[modality]
                else:
                    raise TypeError(
                        f"Modality {modality} is missing in SampleList. "
                        + f"Expected to find 'input_ids' or {modality}."
                    )

                if text_ids.dim() > 2:
                    input_ids[modality] = text_ids[:, idx]
                else:
                    input_ids[modality] = text_ids
            elif self.modality_type[idx] == "image":
                if "image" in sample_list:
                    input_ids[modality] = sample_list["image"]
                # input_modal is originally used by MMBT, added for
                # cross-compatibility of interops and datasets.
                elif "input_modal" in sample_list:
                    input_ids[modality] = sample_list["input_modal"]
                elif "image_feature_0" in sample_list:
                    input_ids[modality] = sample_list["image_feature_0"]
                elif modality in sample_list:
                    input_ids[modality] = sample_list[modality]
                else:
                    raise TypeError(
                        f"Modality {modality} is missing in SampleList. "
                        + "Expected to find 'image', 'input_modal', "
                        + f"'image_feature_0' or {modality}."
                    )
            else:
                if modality in sample_list:
                    input_ids[modality] = sample_list[modality]
                else:
                    # TODO: Later deliberate if missing modalities should
                    # be supported in MMFT.
                    raise TypeError(f"Modality {modality} is missing in SampleList.")

            # In the other case feature will be skipped, as it is not present in
            # the sample list
            if encoder is not None:
                input_ids[modality] = encoder(input_ids[modality])

        # Position IDs
        position_ids: Dict[str, Tensor] = {}
        for modality in self.modality_keys:
            end_idx = input_ids[modality].size(1)
            # In case of dim=2, this is a direct feature, so there
            # is only one element for this modality
            if input_ids[modality].dim() == 2:
                end_idx = 1

            position_ids[modality] = (
                torch.arange(
                    0, end_idx, dtype=torch.long, device=input_ids[modality].device
                )
                .unsqueeze(0)
                .expand((input_ids[modality].size(0), end_idx))
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
                segment_ids[modality] = torch.full(
                    input_ids[modality].size()[:-1],
                    fill_value=self.modality_segments[idx],
                    dtype=torch.long,
                    device=input_ids[modality].device,
                )

            # Should be B X L, if only B in case of direct features, make it B X 1
            if segment_ids[modality].dim() == 1:
                segment_ids[modality] = segment_ids[modality].unsqueeze(dim=1)

        # Masks
        masks: Dict[str, Tensor] = {}
        for idx, modality in enumerate(self.modality_keys):
            if self.modality_type[idx] == "text" and "input_mask" in sample_list:
                if sample_list["input_mask"].dim() > 2:
                    masks[modality] = sample_list["input_mask"][:, idx]
                else:
                    masks[modality] = sample_list["input_mask"]
            else:
                mask_attribute = f"{modality}_mask"
                if mask_attribute in sample_list:
                    masks[modality] = sample_list[mask_attribute]
                else:
                    masks[modality] = torch.ones(
                        input_ids[modality].size()[:-1],
                        dtype=torch.long,
                        device=input_ids[modality].device,
                    )

            # Should be B X L, if only B in case of direct features, make it B X 1
            if masks[modality].dim() == 1:
                masks[modality] = masks[modality].unsqueeze(dim=1)

        return MMFTransformerInput(input_ids, position_ids, segment_ids, masks)

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
