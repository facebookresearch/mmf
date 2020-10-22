# Copyright (c) Facebook, Inc. and its affiliates.
import collections
from copy import deepcopy

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.encoders import MultiModalEncoderBase
from mmf.utils.build import build_classifier_layer
from mmf.utils.modeling import get_bert_configured_parameters


class FusionBase(MultiModalEncoderBase):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def build(self):
        encoders = self._build_encoders(self.config)
        text_encoder, modal_encoder = encoders[0], encoders[1]

        self._modal_encoder_config = self.config.modal_encoder
        self._is_direct_features_input = self.config.direct_features_input
        self._encoder_config = getattr(text_encoder, "config", None)
        self.text = text_encoder
        self.modal = modal_encoder

    def forward(
        self,
        text,
        modal,
        text_args=None,
        modal_args=None,
        text_kwargs=None,
        modal_kwargs=None,
    ):
        if text_args is None:
            text_args = []
        if modal_args is None:
            modal_args = []
        if text_kwargs is None:
            text_kwargs = {}
        if modal_kwargs is None:
            modal_kwargs = {}
        text = self.text(text, *text_args, **text_kwargs)

        # Case of bert encoder, we only need pooled output. For BertModelJIT encoder
        # pooled output is the 2nd in the tuple(sequence, pooled, encoded_layers)
        if isinstance(text, collections.abc.Sequence) and len(text) >= 2:
            text = text[1]

        modal = self.modal(modal, *modal_args, **modal_kwargs)
        modal = torch.flatten(modal, start_dim=1)
        text = torch.flatten(text, start_dim=1)
        return text, modal


@registry.register_model("concat_bert")
class ConcatBERT(BaseModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self._is_direct_features_input = config.direct_features_input

    @classmethod
    def config_path(cls):
        return "configs/models/fusions/concat_bert.yaml"

    def build(self):
        self.base = FusionBase(self.config)
        num_features = self.config.num_features
        if not self._is_direct_features_input:
            num_features = self.config.modal_encoder.params.num_output_features

        # As the in_dim is dynamically calculated we need to copy classifier_config
        classifier_config = deepcopy(self.config.classifier)
        classifier_config.params.in_dim = num_features * self.config.modal_hidden_size
        classifier_config.params.in_dim += self.config.text_hidden_size
        self.classifier = build_classifier_layer(classifier_config)

        if self.config.freeze_text or self.config.freeze_complete_base:
            for p in self.base.text.parameters():
                p.requires_grad = False

        if self.config.freeze_modal or self.config.freeze_complete_base:
            for p in self.base.modal.parameters():
                p.requires_grad = False

    def get_optimizer_parameters(self, config):
        # For finetuning setup, we have classifier
        lr = config.optimizer.params.lr
        model_config = getattr(config.model_config, config.model, {})
        finetune_lr_multiplier = getattr(model_config, "finetune_lr_multiplier", 1)
        # Finetune the bert pretrained part with finetune_lr_multiplier if it is set
        parameters = get_bert_configured_parameters(
            self.base, lr * finetune_lr_multiplier
        )
        parameters += get_bert_configured_parameters(self.classifier, lr)
        return parameters

    def forward(self, sample_list):
        text = sample_list.input_ids
        mask = sample_list.input_mask
        segment = sample_list.segment_ids

        if self._is_direct_features_input:
            modal = sample_list.image_feature_0
        else:
            modal = sample_list.image

        text_embedding, modal_embedding = self.base(text, modal, [mask, segment])
        embedding = torch.cat([text_embedding, modal_embedding], dim=-1)
        output = {}
        output["scores"] = self.classifier(embedding)
        return output


@registry.register_model("concat_bow")
class ConcatBoW(BaseModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self._is_direct_features_input = config.direct_features_input

    @classmethod
    def config_path(cls):
        return "configs/models/fusions/concat_bow.yaml"

    def build(self):
        self.base = FusionBase(self.config)
        num_features = self.config.num_features
        if not self._is_direct_features_input:
            num_features = self.config.modal_encoder.params.num_output_features

        # As the in_dim is dynamically calculated we need to copy classifier_config
        classifier_config = deepcopy(self.config.classifier)
        classifier_config.params.in_dim = num_features * self.config.modal_hidden_size
        classifier_config.params.in_dim += self.config.text_hidden_size
        self.classifier = build_classifier_layer(classifier_config)

    def forward(self, sample_list):
        text = sample_list.text
        if self._is_direct_features_input:
            modal = sample_list.image_feature_0
        else:
            modal = sample_list.image

        text_embedding, modal_embedding = self.base(text, modal)
        embedding = torch.cat([text_embedding, modal_embedding], dim=-1)
        output = {}
        output["scores"] = self.classifier(embedding)
        return output


@registry.register_model("late_fusion")
class LateFusion(BaseModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self._is_direct_features_input = config.direct_features_input

    @classmethod
    def config_path(cls):
        return "configs/models/fusions/late_fusion.yaml"

    def build(self):
        self.base = FusionBase(self.config)
        num_features = self.config.num_features
        if not self._is_direct_features_input:
            num_features = self.config.modal_encoder.params.num_output_features

        # As the in_dim is dynamically calculated we need to copy classifier_config
        modal_classifier_config = deepcopy(self.config.modal_classifier)
        modal_classifier_config.params.in_dim = (
            num_features * self.config.modal_hidden_size
        )
        self.modal_classifier = build_classifier_layer(modal_classifier_config)

        text_classifier_config = deepcopy(self.config.text_classifier)
        text_classifier_config.params.in_dim = self.config.text_hidden_size
        self.text_classifier = build_classifier_layer(text_classifier_config)

    def forward(self, sample_list):
        text = sample_list.input_ids
        mask = sample_list.input_mask
        segment = sample_list.segment_ids

        if self._is_direct_features_input:
            modal = sample_list.image_feature_0
        else:
            modal = sample_list.image

        text_embedding, modal_embedding = self.base(text, modal, [mask, segment])
        text = self.text_classifier(text_embedding)
        modal = self.modal_classifier(modal_embedding)
        output = {}
        output["scores"] = (text + modal) / 2
        return output
