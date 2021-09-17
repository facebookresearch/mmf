# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import logging

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.losses import MMFLoss
from mmf.utils.build import build_encoder
from mmf.utils.modeling import get_bert_configured_parameters
from torch import nn


logger = logging.getLogger()


@registry.register_model("vilt")
class ViLT(BaseModel):
    @classmethod
    def config_path(cls):
        return "configs/models/vilt/defaults.yaml"

    def build(self):
        self.text_embeddings = build_encoder(self.config.text_embeddings)
        self.image_embeddings = build_encoder(self.config.image_embeddings)
        self.encoder = build_encoder(self.config.image_encoder)

        # TODO: Add more classifiers later.
        self.heads = nn.ModuleDict()
        head_configs = self.config.get("heads", {})

        self.tasks = self.config.tasks
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(",")

        for task in self.tasks:
            head_config = head_configs[task]
            head_type = head_config.get("type", "mlp")
            head_class = registry.get_transformer_head_class(head_type)
            self.heads[task] = head_class(head_config)

    def init_losses(self):
        self.losses = nn.ModuleDict()
        loss_configs = self.config.get("losses", {})
        for task in self.tasks:
            if task not in loss_configs:
                logger.warning(
                    f"No loss defined for {task}. Head is expected "
                    + "to return dict with 'losses'"
                )
                continue
            loss_config = loss_configs[task]
            self.losses[task] = MMFLoss(loss_config)

    def forward(self, sample_list):
        text_embedding = self.text_embeddings(sample_list)
        image_embedding = self.image_embeddings(sample_list)

        # Feed through encoder
        embeddings = torch.cat([image_embedding, text_embedding], dim=1)
        attention_mask = self.get_attention_mask(
            sample_list, text_embedding, image_embedding
        )
        sequence, _ = self.encoder(embeddings, attention_mask=attention_mask)
        if sequence.dim() != 3:
            sequence = sequence.unsqueeze(1)

        outputs = self.heads[sample_list.dataset_name](
            sequence, processed_sample_list=sample_list
        )

        if isinstance(outputs, collections.MutableMapping) and "losses" in outputs:
            return outputs

        logits = outputs
        if isinstance(outputs, collections.MutableMapping) and "scores" in outputs:
            logits = outputs["scores"]
        logits = logits.contiguous().view(-1, logits.size(-1))
        output = self.losses[sample_list.dataset_name](sample_list, {"scores": logits})
        return {"losses": output, "scores": logits}

    def get_optimizer_parameters(self, config):
        if hasattr(self.encoder, "get_optimizer_parameters"):
            params = self.encoder.get_optimizer_parameters(config)
        else:
            params = [{"params": self.encoder.parameters()}]
        params += get_bert_configured_parameters(self.text_embeddings)
        params += get_bert_configured_parameters(self.heads)
        params += [{"params": self.image_embeddings.parameters()}]
        return params

    def get_attention_mask(self, sample_list, text_embedding, image_embedding):
        image_mask = getattr(sample_list, "image_mask", None)

        if image_mask is not None and sample_list.input_mask is not None:
            attention_mask = torch.cat((sample_list.input_mask, image_mask), dim=-1)
        elif image_mask is not None:
            text_mask = torch.ones(
                text_embedding.size()[:-1],
                dtype=text_embedding.dtype,
                device=text_embedding.device,
            )
            attention_mask = torch.cat((image_mask, text_mask), dim=-1)
        elif sample_list.input_mask is not None:
            image_mask = torch.ones(
                image_embedding.size()[:-1],
                dtype=image_embedding.dtype,
                device=image_embedding.device,
            )
            attention_mask = torch.cat((image_mask, sample_list.input_mask), dim=-1)
        else:
            attention_mask = None

        if attention_mask is not None:
            attention_mask = attention_mask.masked_fill(
                ~attention_mask.bool(), float("-inf")
            )
            attention_mask = attention_mask[:, None, None, :]

        return attention_mask
