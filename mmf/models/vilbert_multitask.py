# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from transformers.modeling_bert import (
    ACT2FN,
    BertConfig,
    BertLayerNorm,
)

from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import (
    get_optimizer_parameters_for_vilbert_multitask
)

from mmf.models.vilbert import ViLBERTBase


class BertPredictionHeadTransformMultiTask(nn.Module):
    def __init__(self, config, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, hidden_dim)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(hidden_dim, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.classifier(hidden_states)
        return hidden_states


class ViLBERTForMultiTaskClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = ViLBERTBase.from_pretrained(
            self.config.bert_model_name,
            config=BertConfig.from_dict(
                OmegaConf.to_container(self.config, resolve=True)
            ),
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
        )

        self.training_head_type = self.config.training_head_type
        self.num_labels = self.config.num_labels
        self.fusion_method = config.fusion_method

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # This is done to match the implementation with vilbert_multitask codebase
        input_dim = config.bi_hidden_size
        hidden_dim = 2 * input_dim

        self.cls = nn.Linear(input_dim, 2)

        self.vil_prediction = BertPredictionHeadTransformMultiTask(
            self.config, input_dim, hidden_dim, 3129)

        self.vil_prediction_gqa = BertPredictionHeadTransformMultiTask(
            self.config, input_dim, hidden_dim, 1533)

        self.vil_binary_prediction = BertPredictionHeadTransformMultiTask(
            self.config, 2 * input_dim, hidden_dim, 2)

        self.vil_logit = nn.Linear(input_dim, 1)

        self.vil_tri_prediction = nn.Linear(input_dim, 3)

        # for Visual Entailiment tasks
        self.vision_logit = nn.Linear(config.v_hidden_size, 1)
        self.linguisic_logit = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.config.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()

            # Classifier needs to be initialized always as it is task specific
            self.cls.apply(self.bert._init_weights)
            self.vil_prediction.apply(self.bert._init_weights)
            self.vil_prediction_gqa.apply(self.bert._init_weights)
            self.vil_binary_prediction.apply(self.bert._init_weights)
            self.vil_logit.apply(self.bert._init_weights)
            self.vil_tri_prediction.apply(self.bert._init_weights)
            self.vision_logit.apply(self.bert._init_weights)
            self.linguisic_logit.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids,
        image_feature,
        image_location,
        token_type_ids=None,
        attention_mask=None,
        image_attention_mask=None,
        masked_lm_labels=None,
        image_label=None,
        image_target=None,
        next_sentence_label=None,
        output_all_attention_masks=False,
    ):

        (
            sequence_output_t,
            sequence_output_v,
            pooled_output_t,
            pooled_output_v,
            attention_weights,
        ) = self.bert(
            input_ids,
            image_feature,
            image_location,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks,
        )

        all_attention_mask = 0
        vil_prediction = 0
        vil_logit = 0
        vil_binary_prediction = 0
        vision_prediction = 0
        vision_logit = 0
        linguisic_prediction = 0
        linguisic_logit = 0

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            assert False

        vil_binary_prediction = self.cls(pooled_output)

        vil_prediction = self.vil_prediction(pooled_output)
        vil_prediction_gqa = self.vil_prediction_gqa(pooled_output)

        if pooled_output.size(0) % 2 == 0:
            vil_binary_prediction = self.vil_binary_prediction(
                pooled_output.view(-1, pooled_output.size(1) * 2)
            )

        vil_logit = self.vil_logit(pooled_output)
        vil_tri_prediction = self.vil_tri_prediction(pooled_output)
        vision_logit = self.vision_logit(self.dropout(sequence_output_v)) + (
            (1.0 - image_attention_mask) * -10000.0
        ).unsqueeze(2).to(dtype=next(self.parameters()).dtype)
        linguisic_logit = self.linguisic_logit(self.dropout(sequence_output_t))

        return {
            "vil_prediction": vil_prediction,
            "vil_prediction_gqa": vil_prediction_gqa,
            "vil_logit": vil_logit,
            "vil_binary_prediction": vil_binary_prediction,
            "vil_tri_prediction": vil_tri_prediction,
            "vision_prediction": vision_prediction,
            "vision_logit": vision_logit,
            "linguisic_prediction": linguisic_prediction,
            "linguisic_logit": linguisic_logit,
            "all_attention_mask": all_attention_mask,
        }


@registry.register_model("vilbert_multitask")
class ViLBERTMultiTask(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/vilbert_multitask/defaults.yaml"

    # Backward compatibility
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("bert.bert", "model.bert")
            .replace("bert.cls", "model.cls")
            .replace("bert.classifier", "model.classifier")
            .replace("bert.", "model.bert.")
        )

    def build(self):
        self.model = ViLBERTForMultiTaskClassification(self.config)

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def get_image_and_text_features(self, sample_list):

        batch_size = 0
        num_options = 0

        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids
        image_attention_mask = None

        if sample_list.dataset_name == "flickr30k_retrieval":

            image_info = getattr(sample_list, "image_info_0", {})
            image_dim_variable = getattr(image_info, "max_features", None)

            image_label_variable = getattr(sample_list, "image_labels", None)
            if image_label_variable is not None:
                image_label_variable = torch.tensor(
                    image_label_variable, dtype=torch.long
                ).cuda()

            image_feature_variable = getattr(sample_list, "image_feature_0", None)

            # in flickr30k, one training entry can have multiple images, so boxes will
            # have to be pre normalized according to their corresponding image sizes.
            image_location_variable = getattr(image_info, "bbox", None)
            cls_prob = getattr(image_info, "cls_prob", None)
            image_target = np.array(cls_prob, dtype=np.float32)
            image_target_variable = torch.tensor(image_target, dtype=torch.float).cuda()
            image_attention_mask = sample_list.image_mask

            batch_size = image_feature_variable.size(0)
            num_options = bert_input_ids.size(1)

            image_feature_variable = \
                image_feature_variable.view(-1,
                                image_feature_variable.size(2),
                                image_feature_variable.size(3)
                                )
            image_location_variable = \
                image_location_variable.view(-1,
                                    image_location_variable.size(2),
                                    image_location_variable.size(3)
                                    )
            bert_input_ids = bert_input_ids.view(-1, bert_input_ids.size(2))
            bert_input_mask = bert_input_mask.view(-1, bert_input_mask.size(2))
            bert_input_type_ids = bert_input_type_ids.view(-1, bert_input_type_ids.size(2))
            image_attention_mask = image_attention_mask.view(-1, image_attention_mask.size(2))

        else:
            image_info = getattr(sample_list, "image_info_0", {})
            image_dim_variable = getattr(image_info, "max_features", None)
            image_feature_variable = getattr(sample_list, "image_feature_0", None)
            image_label_variable = getattr(sample_list, "image_labels", None)
            if image_label_variable is not None:
                image_label_variable = torch.tensor(
                    image_label_variable, dtype=torch.long
                ).cuda()

            bbox = np.array(getattr(image_info, "bbox", None), dtype=np.float32)
            image_w = np.array(
                getattr(image_info, "image_width", None), dtype=np.float32
            )
            image_h = np.array(
                getattr(image_info, "image_height", None), dtype=np.float32
            )
            image_location = np.zeros(
                (bbox.shape[0], bbox.shape[1], 5), dtype=np.float32
            )
            image_location[:, :, :4] = bbox
            image_location[:, :, 4] = (
                (image_location[:, :, 3] - image_location[:, :, 1])
                * (image_location[:, :, 2] - image_location[:, :, 0])
                / (image_w * image_h)[:, None]
            )
            image_location[:, :, 0] = image_location[:, :, 0] / image_w[:, None]
            image_location[:, :, 1] = image_location[:, :, 1] / image_h[:, None]
            image_location[:, :, 2] = image_location[:, :, 2] / image_w[:, None]
            image_location[:, :, 3] = image_location[:, :, 3] / image_h[:, None]

            image_location_variable = torch.tensor(
                image_location, dtype=torch.float
            ).cuda()

            cls_prob = getattr(image_info, "cls_prob", None)
            image_target = np.array(cls_prob, dtype=np.float32)
            image_target_variable = torch.tensor(image_target, dtype=torch.float).cuda()

        return {
            "input_ids": bert_input_ids,
            "attention_mask": bert_input_mask,
            "token_type_ids": bert_input_type_ids,
            "image_dim": image_dim_variable,
            "image_feature": image_feature_variable,
            "image_location": image_location_variable,
            "image_target": image_target_variable,
            "image_label": image_label_variable,
            "image_attention_mask": image_attention_mask,
            "batch_size": batch_size,
            "num_options": num_options
        }

    def process_output(self, sample_list, params, outputs):

        output = {}
        if sample_list.dataset_name == "flickr30k_retrieval":
            output['scores'] = outputs["vil_logit"].view(params["batch_size"], params["num_options"])

        return output

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_vilbert_multitask(self.model, config)

    def forward(self, sample_list):
        params = self.get_image_and_text_features(sample_list)
        # pretraining labels
        params["masked_lm_labels"] = getattr(sample_list, "lm_label_ids", None)
        # is_random_next = getattr(sample_list, "is_correct", None)
        # TODO(aps): Fix on dataset side
        # params["is_random_next"] = None

        # Prepare Mask
        if params['image_attention_mask'] is None:
            if params["image_feature"] is not None and params["image_dim"] is not None:
                image_mask = (
                    torch.arange(params["image_feature"].size(-2))
                    .expand(*params["image_feature"].size()[:-1])
                    .cuda()
                )
                if len(params["image_dim"].size()) < len(image_mask.size()):
                    params["image_dim"] = params["image_dim"].unsqueeze(-1)
                    assert len(params["image_dim"].size()) == len(image_mask.size())
                image_mask = image_mask < params["image_dim"]
                params["image_attention_mask"] = image_mask.long()
            else:
                params["image_attention_mask"] = None
        params.pop("image_dim")

        output = self.model(
            params["input_ids"],
            params["image_feature"],
            params["image_location"],
            params["token_type_ids"],
            params["attention_mask"],
            params["image_attention_mask"],
            params["masked_lm_labels"],
            params["image_label"],
            params["image_target"],
        )

        output = self.process_output(sample_list, params, output)

        return output
