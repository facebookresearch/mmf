# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy

import torch

from mmf.common.registry import registry
from mmf.models.base_model import BaseModel

from mmf.utils.build import (
    build_text_encoder,
    build_image_encoder,
    build_classifier_layer
)
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)

@registry.register_model("align_base")
class AlignBase(BaseModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        self.final_hidden_size = config.get("final_hidden_size", 512)
        self.norm_embeddings = self.config.get("norm_embeddings", False)

        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/unimodal_alignment/default.yaml"

    def build(self):
        self.image_encoder = build_image_encoder(self.config.image_encoder, direct_features=False)
        self.text_encoder = build_text_encoder(self.config.text_encoder)

        image_proj_config = deepcopy(self.config.image_projection)
        self.image_proj = build_classifier_layer(image_proj_config)

        text_proj_config = deepcopy(self.config.text_projection)
        self.text_proj = build_classifier_layer(text_proj_config)

    def forward(self, sample_list):
        text = transform_to_batch_sequence(sample_list.input_ids)
        mask = transform_to_batch_sequence(sample_list.input_mask)
        segment = transform_to_batch_sequence(sample_list.segment_ids)
        image = sample_list.image

        text = self.text_encoder(text, mask, segment)

        # Case of bert encoder, we only need pooled output
        if len(text) == 2:
            text = text[1]

        image = self.image_encoder(image)

        image_emb = torch.flatten(image, start_dim=1)
        text_emb = torch.flatten(text, start_dim=1)

        # Project embeddings to the same dimension
        image_proj = self.image_proj(image_emb)
        text_proj = self.text_proj(text_emb)

        if self.norm_embeddings:
            image_proj = torch.nn.functional.normalize(image_proj, 2, dim=1)
            text_proj = torch.nn.functional.normalize(text_proj, 2, dim=1)

        output = {
            "scores": image_proj,
            "targets": text_proj
        }

        return output
