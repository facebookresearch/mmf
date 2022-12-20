# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from copy import deepcopy
from typing import Dict

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.models.unit.unit_base_model import (
    AttributeHead,
    build_detection_loss,
    MLP,
    UniTBaseModel,
)
from mmf.modules.encoders import TransformerEncoder
from mmf.utils.distributed import byte_tensor_to_object
from torch import nn, Tensor

try:
    from transformers3.modeling_bert import BertPredictionHeadTransform
except ImportError:
    from transformers.modeling_bert import BertPredictionHeadTransform


logger = logging.getLogger(__name__)


@registry.register_model("unit")
class UniT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "configs/models/unit/defaults.yaml"

    # Backward compatibility for code from older mmbt
    @classmethod
    def format_state_key(cls, key):
        return key.replace("detr_model.", "unit_base_model.")

    def build(self):
        # build the base model (based on DETR)
        self.unit_base_model = UniTBaseModel(self.config.base_args)

        def keep_only_backbone_params(model_state_dict):
            keys = list(model_state_dict.keys())
            for k in keys:
                if "backbone" not in k:
                    model_state_dict.pop(k)

        ckpt_path = self.config.base_ckpt_path
        if ckpt_path != "":
            logger.info(f"initializing base model (UniT) from {ckpt_path}")
            if ckpt_path.startswith("https"):
                base_checkpoint = torch.hub.load_state_dict_from_url(
                    ckpt_path, check_hash=True
                )
            else:
                base_checkpoint = torch.load(ckpt_path)
            if self.config.base_ckpt_load_backbone_only:
                keep_only_backbone_params(base_checkpoint["model"])
                self.unit_base_model.load_state_dict(
                    base_checkpoint["model"], strict=False
                )
            else:
                self.unit_base_model.load_state_dict(
                    base_checkpoint["model"], strict=True
                )

        # build the text encoder (BERT)
        self.bert_model = TransformerEncoder(self.config.base_args.bert_config)
        detr_hidden_dim = self.config.base_args.decoder_hidden_dim
        bert_config = deepcopy(self.bert_model.config)
        self.bert_projection = nn.Linear(bert_config.hidden_size, detr_hidden_dim)
        self.bert_pos_projection = nn.Linear(bert_config.hidden_size, detr_hidden_dim)

        self.classifiers = nn.ModuleDict()

        self.task_embeddings_lang = nn.Identity()
        if self.config.base_args.use_task_embedding_in_lang_encoder:
            self.task_embeddings_lang = nn.Embedding(
                self.config.max_task_num, bert_config.hidden_size
            )

        bert_config.hidden_size = detr_hidden_dim

        # build the task-specific output heads
        self.class_embeds = nn.ModuleDict()
        self.bbox_embeds = nn.ModuleDict()
        self.det_losses = nn.ModuleDict()
        for dataset_name in self.config.base_args.num_queries.get("detection", []):
            num_cls = self.config.heads["detection"][dataset_name]["num_classes"]
            self.class_embeds[dataset_name] = nn.Linear(detr_hidden_dim, num_cls + 1)
            self.bbox_embeds[dataset_name] = MLP(detr_hidden_dim, detr_hidden_dim, 4, 3)
            attr_head = None
            if self.config.heads["detection"][dataset_name]["use_attr"]:
                attr_head = AttributeHead(
                    num_cls, self.config.base_args.attribute_class_num, detr_hidden_dim
                )
            self.det_losses[dataset_name] = build_detection_loss(
                self.config.base_args, num_cls, attr_head
            )

        vl_classifiers = nn.ModuleDict()
        for dataset_name in self.config.base_args.num_queries.get("vl", []):
            vl_classifiers[dataset_name] = nn.Sequential(
                BertPredictionHeadTransform(bert_config),
                nn.Linear(
                    bert_config.hidden_size,
                    self.config.heads["vl"][dataset_name]["num_labels"],
                ),
            )

        self.classifiers["vl"] = vl_classifiers
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        glue_classifiers = nn.ModuleDict()
        for dataset_name in self.config.base_args.num_queries.get("glue", []):
            glue_classifiers[dataset_name] = nn.Sequential(
                BertPredictionHeadTransform(bert_config),
                nn.Linear(
                    bert_config.hidden_size,
                    self.config.heads["glue"][dataset_name]["num_labels"],
                ),
            )

        self.classifiers["glue"] = glue_classifiers

        self.loss_calculation_fn = {}
        self.loss_calculation_fn["detection"] = self.detection_loss_calculation
        self.loss_calculation_fn["vl"] = self.classifier_loss_calculation
        self.loss_calculation_fn["glue"] = self.classifier_loss_calculation

        self.losses_dict = {}
        self.losses_dict["vl"] = {
            name: self.get_loss_fn(self.config.heads["vl"][name]["loss_type"])
            for name in self.config.heads["vl"]
        }
        self.losses_dict["glue"] = {
            name: self.get_loss_fn(self.config.heads["glue"][name]["loss_type"])
            for name in self.config.heads["glue"]
        }

    def forward_bert_with_task_idx(self, sample_list):
        bert = self.bert_model.module
        input_ids = sample_list.input_ids
        attention_mask = sample_list.input_mask
        token_type_ids = sample_list.segment_ids
        device = input_ids.device

        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=device)

        input_shape = input_ids.size()

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        start_idx = 0
        if self.config.base_args.use_task_embedding_in_lang_encoder:
            bs = input_ids.size(0)
            task_idx = self.get_task_idx(sample_list.dataset_name)
            task_embed = self.task_embeddings_lang.weight[task_idx]
            task_embed = task_embed.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1)
            embedding_output = torch.cat([task_embed, embedding_output], dim=1)
            task_attention_mask = embedding_output.new_ones((bs, 1))
            attention_mask = torch.cat([task_attention_mask, attention_mask], dim=1)
            start_idx = 1

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None for _ in range(bert.config.num_hidden_layers)]
        encoder_outputs = bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )
        sequence_output = encoder_outputs[0][:, start_idx:, :]
        pos_embeddings = self.bert_model.embeddings.position_embeddings(position_ids)

        return sequence_output, pos_embeddings

    def forward(self, sample_list):
        detr_outputs = {}
        task_type = self.get_task_type(sample_list.dataset_name)
        text_src = None
        text_mask = None
        text_pos = None
        img_src = None
        if task_type == "vl" or task_type == "glue":
            if task_type == "vl":
                img_src = sample_list.image
            text_src, pos_embeddings = self.forward_bert_with_task_idx(sample_list)
            # 768 -> 256
            text_src = self.bert_projection(text_src)
            text_pos = self.bert_pos_projection(pos_embeddings)

            text_mask = sample_list.input_mask
            if self.config.keep_only_bert_cls[task_type][sample_list.dataset_name]:
                # take only the [CLS] token's hidden state
                text_src = text_src[:, 0:1, :]
                text_pos = text_pos[0:1, :]
                text_mask = text_mask[:, 0:1]
        elif task_type == "detection":
            img_src = sample_list.image

        detr_outputs = self.unit_base_model(
            img_src=img_src,
            text_src=text_src,
            text_mask=text_mask,
            text_pos=text_pos,
            task_type=task_type,
            dataset_name=sample_list.dataset_name,
            task_idx=self.get_task_idx(sample_list.dataset_name),
        )

        output = self.loss_calculation_fn[task_type](detr_outputs, sample_list)
        return output

    def detection_loss_calculation(self, detr_outputs: Dict[str, Tensor], sample_list):
        hs = detr_outputs["hidden_states"]

        outputs_class = self.class_embeds[sample_list.dataset_name](hs)
        outputs_coord = self.bbox_embeds[sample_list.dataset_name](hs).sigmoid()
        detr_outputs.update(
            {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
                "hs_for_attr": hs[-1],
            }
        )
        # skip loss computation on test set (which usually doesn't contain labels)
        if sample_list.dataset_type != "test":
            if self.config.base_args.aux_loss:
                detr_outputs["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b, "hs_for_attr": c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], hs[:-1])
                ]

            criterion = self.det_losses[sample_list.dataset_name]
            targets = [byte_tensor_to_object(t) for t in sample_list.targets_enc]
            targets = [{k: v.to(hs.device) for k, v in t.items()} for t in targets]
            sample_list.targets = targets
            loss_dict = criterion(detr_outputs, sample_list.targets)
            weight_dict = criterion.weight_dict
            loss_prefix = f"{sample_list.dataset_type}/{sample_list.dataset_name}/"
            losses = {
                (loss_prefix + f"{k}"): loss_dict[k]
                * weight_dict[k]
                * self.config.detection_loss_weight
                for k in loss_dict.keys()
                if k in weight_dict
            }
            detr_outputs["losses"] = losses

        if (
            self.config.heads["detection"][sample_list.dataset_name]["use_attr"]
            and self.config.predict_attributes
        ):
            hs_for_attr = detr_outputs["hs_for_attr"]
            top_obj_class = detr_outputs["pred_logits"][..., :-1].argmax(dim=-1)
            attr_head = self.det_losses[sample_list.dataset_name].attribute_head
            detr_outputs["attr_logits"] = attr_head(hs_for_attr, top_obj_class)

        return detr_outputs

    def classifier_loss_calculation(self, detr_outputs: Dict[str, Tensor], sample_list):
        task_type = self.get_task_type(sample_list.dataset_name)
        hs = detr_outputs["hidden_states"]
        if not self.config.loss_on_all_hs:
            hs = detr_outputs["hidden_states"][-1:]
        num_queries = self.config.base_args.num_queries[task_type][
            sample_list.dataset_name
        ]
        assert hs[0].size(1) == num_queries
        losses = {}
        scores = None
        detr_outputs = {}
        num_labels = self.config.heads[task_type][sample_list.dataset_name][
            "num_labels"
        ]

        for idx, current_hs in enumerate(hs):
            pooled_output = current_hs[:, -num_queries, :]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifiers[task_type][sample_list.dataset_name](
                pooled_output
            )
            reshaped_logits = logits.contiguous().view(-1, num_labels)
            scores = reshaped_logits
            # skip loss computation on test set (which usually doesn't contain labels)
            if sample_list.dataset_type != "test":
                loss_prefix = f"{sample_list.dataset_type}/{sample_list.dataset_name}/"
                loss = self.losses_dict[task_type][sample_list.dataset_name](
                    scores, sample_list.targets
                )
                if sample_list.dataset_name == "vqa2":
                    loss *= sample_list.targets.size(1)
                losses[loss_prefix + f"loss_{idx}"] = loss

        detr_outputs["scores"] = scores
        detr_outputs["losses"] = losses
        return detr_outputs

    def get_optimizer_parameters(self, config):
        detr_params = [
            {
                "params": [
                    p
                    for n, p in self.unit_base_model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.unit_base_model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.config.base_args.lr_backbone,
            },
        ]

        vqa_params = [
            {"params": self.bert_model.parameters()},
            {"params": self.bert_projection.parameters()},
            {"params": self.task_embeddings_lang.parameters()},
            {"params": self.bert_pos_projection.parameters()},
            {"params": self.classifiers.parameters()},
            {"params": self.class_embeds.parameters()},
            {"params": self.bbox_embeds.parameters()},
            {"params": self.det_losses.parameters()},
        ]
        return vqa_params + detr_params

    def get_task_idx(self, dataset_name):
        task_type = self.get_task_type(dataset_name)
        assert task_type in self.config.heads
        return self.config.heads[task_type][dataset_name]["task_idx"]

    def get_task_type(self, dataset_name):
        task_type = "detection"
        if dataset_name in self.config.heads["vl"]:
            task_type = "vl"
        elif dataset_name in self.config.heads["glue"]:
            task_type = "glue"
        return task_type

    def get_loss_fn(self, loss_type):
        if loss_type == "binary_cross_entropy_with_logits":
            return nn.functional.binary_cross_entropy_with_logits
        elif loss_type == "cross_entropy":
            return nn.functional.cross_entropy
        else:
            raise Exception(f"Unknown loss type: {loss_type}")
