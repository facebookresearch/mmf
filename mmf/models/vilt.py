# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import dataclass, field

import torch
from mmf.modules.encoders import TransformerEncoder, ViTEncoder
from torch import nn


class ViLTImageEmbedding(nn.Module):
    """
    Patch embedding used for ViLT.
    https://arxiv.org/pdf/2102.03334.pdf
    Implementation based off
    https://github.com/dandelin/ViLT/blob/master/vilt/modules/vilt_module.py
    Using huggingface ViT modules.
    Can be built with random init or the embeddings weights from an exisiting
    ViT model from huggingface. Model list: availible at
    https://huggingface.co/models?other=vit&sort=downloads
    """

    @dataclass
    class Config:
        image_size: list = field(default_factory=lambda: [224, 224])
        hidden_dropout_prob: float = 0
        hidden_size: int = 768
        patch_size: int = 16
        num_channels: int = 3
        random_init: bool = True
        pretrained_model_name: str = "google/vit-base-patch16-224"

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = ViTEncoder(self.config).embeddings
        self.token_type_embeddings = nn.Embedding(1, self.config.hidden_size)

    def forward(self, image):
        if image.dim() == 5:
            # manual collation for SimCLR inputs (when VISSL collator is not used)
            # make num_view the 1st dimension to be consistent with VISSL SimCLR
            image = image.permute(1, 0, 2, 3, 4).flatten(start_dim=0, end_dim=1)

        img_embeddings = self.embedding(image)

        img_segment_ids = torch.zeros(
            img_embeddings.size()[:-1],
            dtype=img_embeddings.dtype,
            device=img_embeddings.device,
        ).long()
        img_type_embed = self.token_type_embeddings(img_segment_ids)
        img_embeddings = img_embeddings + img_type_embed
        return img_embeddings


class ViLTTextEmbedding(nn.Module):
    @dataclass
    class Config:
        hidden_dim: int = 768
        hidden_size: int = 768
        bert_model_name: str = "bert-base-uncased"

    def __init__(self, config: Config):

        super().__init__()
        self.config = config
        text_encoder = TransformerEncoder(self.config)
        self.text_embeddings = text_encoder.embeddings
        encoder_output_dim = self.config.hidden_dim
        self.text_projection = nn.Linear(
            text_encoder.config.hidden_size, encoder_output_dim
        )

    def forward(self, input_ids, segment_ids):
        text_embedding = self.text_embeddings(input_ids, token_type_ids=segment_ids)
        text_embedding = self.text_projection(text_embedding)
        return text_embedding
