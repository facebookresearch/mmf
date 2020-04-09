import os
from copy import deepcopy

import torch
from torch import nn
from transformers.configuration_mmbt import MMBTConfig
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.modeling_auto import AutoModelForPreTraining
from transformers.modeling_bert import BertPredictionHeadTransform
from transformers.modeling_mmbt import MMBTModel

from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.modules.encoders import MultiModalEncoderBase
from pythia.utils.modeling import get_optimizer_parameters_for_bert


class MMBTBase(MultiModalEncoderBase):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def build(self):
        encoders = self._build_encoders(self.config)
        text_encoder, modal_encoder = encoders[0], encoders[1]
        self._encoder_config = text_encoder.config

        self._mmbt_config = MMBTConfig(
            self._encoder_config,
            num_labels=self.config.num_labels,
            modal_hidden_size=self.config.modal_hidden_size,
        )

        self.mmbt = MMBTModel(self._mmbt_config, text_encoder, modal_encoder)

    def forward(self, sample_list):
        if self._is_direct_features_input:
            input_modal = sample_list.image_feature_0
        else:
            input_modal = sample_list.image

        modal_start_token = None
        if self.config.use_modal_start_token:
            modal_start_token = sample_list.input_ids[:, 0].clone().detach()

        modal_end_token = None
        if self.config.use_modal_end_token:
            modal_end_token = sample_list.input_ids[:, -1].clone().detach()

        # See details of inputs at
        # https://github.com/huggingface/transformers/blob/1789c7/src/transformers/modeling_mmbt.py#L101
        output = self.mmbt(
            input_modal,
            input_ids=sample_list.input_ids,
            modal_start_tokens=modal_start_token,
            modal_end_tokens=modal_end_token,
            attention_mask=sample_list.input_mask,
            token_type_ids=sample_list.segment_ids,
            modal_token_type_ids=None,
            position_ids=None,
            modal_position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        )

        return output


class MMBTForPreTraining(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.bert = MMBTBase(config, *args, **kwargs)
        self.encoder_config = self.bert.encoder_config

        pretraining_module = AutoModelForPreTraining.from_pretrained(
            self.config.bert_model_name,
            config=self.encoder_config,
            cache_dir=os.path.join(
                str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed_{}".format(-1)
            ),
        )

        self.cls = deepcopy(pretraining_module.cls)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        if hasattr(self, "cls"):
            self.bert.mmbt.transformer._tie_or_clone_weights(
                self.cls.predictions.decoder,
                self.bert.mmbt.transformer.embeddings.word_embeddings,
            )

    def forward(self, sample_list):
        module_output = self.bert(sample_list)
        sequence_output, pooled_output = module_output[0], module_output[1]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        output = {}
        if (
            self.encoder_config.output_hidden_states
            or self.encoder_config.output_attentions
        ):
            output["extras"] = module_output[2:]

        loss_key = "{}/{}".format(sample_list.dataset_name, sample_list.dataset_type)

        if "lm_label_ids" in sample_list and sample_list.lm_label_ids is not None:
            output["logits"] = prediction_scores
            lm_label_ids = sample_list.lm_label_ids
            # Only take last scores which are text's scores and ignore image scores
            text_scores = (
                prediction_scores[:, -lm_label_ids.size(1) :]
                .contiguous()
                .view(-1, self.encoder_config.vocab_size)
            )
            masked_lm_loss = self.loss_fct(
                text_scores, sample_list.lm_label_ids.contiguous().view(-1)
            )
            output["losses"] = {}
            output["losses"]["{}/masked_lm_loss".format(loss_key)] = masked_lm_loss

        # Add alignment loss if present
        if (
            "image_text_alignment" in sample_list
            and sample_list.image_text_alignment is not None
        ):
            output["seq_relationship_logits"] = seq_relationship_score
            alignment_loss = self.loss_fct(
                seq_relationship_score.contiguous().view(-1),
                sample_list.image_text_alignment.contiguous().view(-1),
            )
            output["losses"]["{}/alignment_loss".format(loss_key)] = alignment_loss

        return output


class MMBTForClassification(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.bert = MMBTBase(config, *args, **kwargs)
        self.encoder_config = self.bert.encoder_config

        self.dropout = nn.Dropout(self.encoder_config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.encoder_config),
            nn.Linear(self.encoder_config.hidden_size, self.config.num_labels),
        )

    def forward(self, sample_list):
        module_output = self.bert(sample_list)
        pooled_output = module_output[1]
        output = {}

        if (
            self.encoder_config.output_hidden_states
            or self.encoder_config.output_attentions
        ):
            output["extras"] = module_output[2:]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.config.num_labels)
        output["scores"] = reshaped_logits

        return output


@registry.register_model("mmbt")
class MMBT(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.base = MMBTForPreTraining(self.config)
        else:
            self.base = MMBTForClassification(self.config)

        if self.config.freeze_complete_base or self.config.freeze_text:
            for p in self.base.bert.mmbt.transformer.parameters():
                p.requires_grad = False

        if self.config.freeze_complete_base or self.config.freeze_modal:
            for p in self.base.bert.mmbt.modal_encoder.parameters():
                p.requires_grad = False

    @classmethod
    def config_path(cls):
        return "configs/models/mmbt/pretrain.yaml"

    def forward(self, sample_list):
        return self.base(sample_list)

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.base, config)
