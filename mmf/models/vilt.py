# Copyright (c) Facebook, Inc. and its affiliates.

from dataclasses import asdict, dataclass, field

import torch
from mmf.modules.encoders import TransformerEncoder, ViTEncoder
from omegaconf import OmegaConf
from torch import Tensor, nn


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
        hidden_dim: int = 768
        patch_size: int = 16
        num_channels: int = 3
        random_init: bool = True
        pretrained_model_name: str = "google/vit-base-patch16-224"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = OmegaConf.create({**asdict(self.Config()), **kwargs})
        self.embedding = ViTEncoder(self.config).embeddings
        self.token_type_embeddings = nn.Embedding(2, self.config.hidden_dim)

    def forward(self, image: Tensor) -> Tensor:
        if image.dim() == 5:
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

    def __init__(self, *args, **kwargs):

        super().__init__()
        self.config = OmegaConf.create({**asdict(self.Config()), **kwargs})
        text_encoder = TransformerEncoder(self.config)
        self.text_embeddings = text_encoder.embeddings
        self.token_type_embeddings = nn.Embedding(2, self.config.hidden_dim)

    def forward(self, input_ids: Tensor, segment_ids: Tensor) -> Tensor:
        text_embedding = self.text_embeddings(input_ids, token_type_ids=segment_ids)
        # official vilt repo adds type embeddings twice, once in the bert embeddings
        # and a seperate time directly
        text_type_embed = self.token_type_embeddings(segment_ids)
        return text_embedding + text_type_embed
