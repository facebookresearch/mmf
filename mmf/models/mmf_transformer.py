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
from transformers.modeling_bert import (
    BertOnlyMLMHead,
    BertPooler,
    BertPredictionHeadTransform,
)


@dataclass
class MMFTransformerModalityConfig(BaseTransformerModalityConfig):
    pass


@dataclass
class MMFTransformerBackendConfig(BaseTransformerBackendConfig):
    pass


class MMFTransformerInput(BaseTransformerInput):
    input_ids: Dict[str, Tensor]  # dict of input ids for all modalities
    position_ids: Dict[str, Tensor]  # dict of position ids for all modalities
    segment_ids: Dict[str, Tensor]  # dict of segment/token type ids for all modalities
    masks: Dict[str, Tensor]  # dict of masks for all modalities
    mlm_labels_list: List[Tensor]  # list of text token masks for all modalities


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
        self.training_head_type = self.config.training_head_type
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
        if self.config.training_head_type == "classification":
            self.pooler = BertPooler(transformer_config)
            self.classifier = nn.Sequential(
                nn.Dropout(transformer_config.hidden_dropout_prob),
                BertPredictionHeadTransform(transformer_config),
                nn.Linear(transformer_config.hidden_size, self.config.num_labels),
            )
        elif self.config.training_head_type == "pretraining":
            self.cls = BertOnlyMLMHead(transformer_config)
            self.vocab_size = transformer_config.vocab_size

    def build_losses(self):
        if self.config.training_head_type == "pretraining":
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def preprocess_sample(self, sample_list: Dict[str, Tensor]) -> BaseTransformerInput:
        """Preprocess the sample list elements and form a BaseTransformerInput
        type object. This object standardizes how we represent multiple modalities.
        Check the definition of this dataclass in BaseTransformer.
        """

        input_ids = self._infer_input_ids(sample_list)
        position_ids = self._infer_position_ids(input_ids)
        masks = self._infer_masks(sample_list, input_ids)
        segment_ids = self._infer_segment_ids(sample_list, input_ids)
        mlm_labels_list = self._infer_mlm_labels(sample_list, input_ids)

        return MMFTransformerInput(
            input_ids, position_ids, segment_ids, masks, mlm_labels_list
        )

    def _infer_input_ids(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Input IDs (or text tokens/image features)
        input_ids: Dict[str, Tensor] = {}
        current_text_idx = 0
        for idx, encoder in enumerate(self.encoders.values()):
            modality = self.modality_keys[idx]
            if self.modality_type[idx] == "text":
                # First, check if standard input_ids corresponds to text
                # if not, check for modality key inside the sample list
                text_ids = self._check_keys_for_modality(
                    sample_list, ("input_ids", modality)
                )
                # This handles the case of more than one text modalities
                # with type text. The input_ids must be stacked in this case.
                # For example, if there are two text modalities, input ids should
                # look have shape: B X 2 X L where second dim points to stacked
                # text ids. Furthermore, make sure that the sequence of modalities
                # in config is same as the sequence in the stacked input ids.
                if text_ids.dim() > 2:
                    input_ids[modality] = text_ids[:, current_text_idx]
                    current_text_idx += 1
                else:
                    input_ids[modality] = text_ids
            elif self.modality_type[idx] == "image":
                # input_modal is originally used by MMBT, added for
                # cross-compatibility of interops and datasets.
                input_ids[modality] = self._check_keys_for_modality(
                    sample_list, ("image", "input_modal", "image_feature_0", modality)
                )
            else:
                # TODO: Later deliberate if missing modalities should
                # be supported in MMFT.
                input_ids[modality] = self._check_keys_for_modality(
                    sample_list, (modality,)
                )
            # In the other case feature will be skipped, as it is not present in
            # the sample list
            if encoder is not None:
                input_ids[modality] = encoder(input_ids[modality])

            # For a feature which is of shape B X D and
            # is not text (which is B X L converted later by embeddings to B X L X D)
            # We convert it to B X 1 X D to signify single position dim.
            if self.modality_type[idx] != "text" and input_ids[modality].dim() == 2:
                input_ids[modality] = input_ids[modality].unsqueeze(1)

        return input_ids

    def _check_keys_for_modality(
        self, sample_list: Dict[str, Tensor], keys: List[str]
    ) -> Tensor:
        assert len(keys) != 0

        for key in keys:
            if key in sample_list:
                return sample_list[key]

        # Reaching here means nothing was found.
        # Easier to write code this way to keep torchscript happy
        if len(keys) == 1:
            expected_list = keys[0]
        else:
            expected_list: str = ", ".join(keys[:-1])
            expected_list = f"{expected_list} or {keys[-1]}"
        raise TypeError(
            f"Missing modality in SampleList. Expected to find {expected_list}"
        )

    def _infer_position_ids(self, input_ids: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Position IDs
        position_ids: Dict[str, Tensor] = {}
        for modality in self.modality_keys:
            end_idx = input_ids[modality].size(1)
            position_ids[modality] = (
                torch.arange(
                    0, end_idx, dtype=torch.long, device=input_ids[modality].device
                )
                .unsqueeze(0)
                .expand((input_ids[modality].size(0), end_idx))
            )
        return position_ids

    def _infer_masks(
        self, sample_list: Dict[str, Tensor], input_ids: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        masks: Dict[str, Tensor] = {}
        current_text_idx = 0
        for idx, modality in enumerate(self.modality_keys):
            if self.modality_type[idx] == "text" and "input_mask" in sample_list:
                if sample_list["input_mask"].dim() > 2:
                    masks[modality] = sample_list["input_mask"][:, current_text_idx]
                    current_text_idx += 1
                else:
                    masks[modality] = sample_list["input_mask"]
            else:
                mask_attribute = f"{modality}_mask"
                if mask_attribute in sample_list:
                    masks[modality] = sample_list[mask_attribute]
                else:
                    masks[modality] = torch.ones(
                        input_ids[modality].size()[:2],
                        dtype=torch.long,
                        device=input_ids[modality].device,
                    )

        return masks

    def _infer_segment_ids(
        self, sample_list: Dict[str, Tensor], input_ids: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        # Segment IDs
        segment_ids: Dict[str, Tensor] = {}
        current_text_idx = 0
        for idx, modality in enumerate(self.modality_keys):
            if self.modality_segments[idx] == -1:
                continue
            if self.modality_type[idx] == "text" and "segment_ids" in sample_list:
                if sample_list["segment_ids"].dim() > 2:
                    segment_ids[modality] = sample_list["segment_ids"][
                        :, current_text_idx
                    ]
                    current_text_idx += 1
                else:
                    segment_ids[modality] = sample_list["segment_ids"]
            else:
                segment_ids[modality] = torch.full(
                    input_ids[modality].size()[:2],
                    fill_value=self.modality_segments[idx],
                    dtype=torch.long,
                    device=input_ids[modality].device,
                )

        return segment_ids

    def _infer_mlm_labels(
        self, sample_list: Dict[str, Tensor], input_ids: Dict[str, Tensor]
    ) -> List[Tensor]:
        # MLM Labels
        mlm_labels: Dict[str, Tensor] = {}
        current_text_idx = 0
        for idx, modality in enumerate(self.modality_keys):
            if self.modality_type[idx] == "text" and "lm_label_ids" in sample_list:
                if sample_list["lm_label_ids"].dim() > 2:
                    mlm_labels[modality] = sample_list["lm_label_ids"][
                        :, current_text_idx
                    ]
                    current_text_idx += 1
                else:
                    mlm_labels[modality] = sample_list["lm_label_ids"]
            else:
                mlm_labels[modality] = torch.full(
                    input_ids[modality].size()[:2],
                    fill_value=-1,
                    dtype=torch.long,
                    device=input_ids[modality].device,
                )

        mlm_labels_list = []
        for modality in self.modality_keys:
            mlm_labels_list.append(mlm_labels[modality])

        return mlm_labels_list

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Sample preprocess
        processed_sample_list = self.preprocess_sample(sample_list)

        # Arrange masks in a list
        masks = []
        for modality in self.modality_keys:
            masks.append(processed_sample_list.masks[modality])

        # Call transformer backend
        sequence_output, encoded_layers = self.backend(
            processed_sample_list.input_ids,
            processed_sample_list.position_ids,
            processed_sample_list.segment_ids,
            masks,
        )

        # Transformer Heads
        output_dict = {}
        if self.training_head_type == "classification":
            pooled_output = self.pooler(sequence_output)
            prediction = self.classifier(pooled_output)
            output_dict = self.postprocess_output(prediction, processed_sample_list)
        elif not torch.jit.is_scripting() and self.training_head_type == "pretraining":
            if not torch.jit.is_scripting():
                prediction_score = self.cls(sequence_output)
                output_dict = self.postprocess_output(
                    prediction_score, processed_sample_list
                )
            else:
                raise AssertionError(
                    "MMF Tranformer : Torchscript is not supported for "
                    + "pretraining mode."
                )

        return output_dict

    def postprocess_output(
        self, prediction: Tensor, processed_sample_list: MMFTransformerInput
    ) -> Dict[str, Tensor]:
        """Postprocess the output from the classifier head and reshape it.
        This will be used to calculate losses and metrics in mmf.
        """
        output_dict = {}
        if self.training_head_type == "classification":
            output_dict["scores"] = prediction.contiguous().view(-1, self.num_labels)
        elif self.training_head_type == "pretraining":
            if not torch.jit.is_scripting():
                output_dict["logits"] = prediction
                masked_labels = torch.cat(processed_sample_list.mlm_labels_list, dim=-1)
                masked_lm_loss = self.ce_loss(
                    prediction.contiguous().view(-1, self.vocab_size),
                    masked_labels.contiguous().view(-1),
                )
                output_dict["losses"] = {}
                output_dict["losses"]["masked_lm_loss"] = masked_lm_loss
            else:
                raise AssertionError(
                    "MMF Transformer pretraining mode cannot be scripted"
                )
        return output_dict
