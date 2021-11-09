# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Optional

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

    def __init__(
        self,
        random_init: bool = True,
        pretrained_model_name: str = "google/vit-base-patch16-224",
        image_size: Optional[List] = None,
        hidden_dropout_prob: Optional[float] = None,
        hidden_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        num_channels: Optional[int] = None,
        *args,
        **kwargs
    ):
        super().__init__()
        config = OmegaConf.create(
            {"random_init": random_init, "pretrained_model_name": pretrained_model_name}
        )
        if image_size is not None:
            config.image_size = image_size
        if hidden_dropout_prob is not None:
            config.hidden_dropout_prob = hidden_dropout_prob
        if hidden_size is not None:
            config.hidden_size = hidden_size
        if patch_size is not None:
            config.patch_size = patch_size
        if num_channels is not None:
            config.num_channels = num_channels

        encoder = ViTEncoder(config)
        self.embedding = encoder.embeddings
        hidden_size = encoder.hf_config.hidden_size
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

    def forward(self, image: Tensor) -> Tensor:
        if image.dim() == 5:
            image = image.permute(1, 0, 2, 3, 4).flatten(start_dim=0, end_dim=1)

        img_embeddings = self.embedding(image)

        img_segment_ids = torch.ones(
            img_embeddings.size()[:-1],
            dtype=img_embeddings.dtype,
            device=img_embeddings.device,
        ).long()
        img_type_embed = self.token_type_embeddings(img_segment_ids)
        img_embeddings = img_embeddings + img_type_embed
        return img_embeddings


class ViLTTextEmbedding(nn.Module):
    def __init__(
        self,
        random_init: bool = True,
        bert_model_name: str = "bert-base-uncased",
        hidden_size: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        *args,
        **kwargs
    ):

        super().__init__()
        config = OmegaConf.create(
            {"bert_model_name": bert_model_name, "random_init": random_init}
        )
        if hidden_size is not None:
            config.hidden_size = hidden_size
        if max_position_embeddings is not None:
            config.max_position_embeddings = max_position_embeddings

        text_encoder = TransformerEncoder(config)
        self.text_embeddings = text_encoder.embeddings
        # the hidden_size param enables hidden_size overrides
        # if hidden_size is None, hidden_size is loaded
        # from the default hf config for the model
        # the actual size of the embeddings will always be in the encoder configs
        hidden_size = text_encoder.config.hidden_size
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

    def forward(self, input_ids: Tensor, segment_ids: Tensor) -> Tensor:
        text_embedding = self.text_embeddings(input_ids, token_type_ids=segment_ids)
        # official vilt repo adds type embeddings twice, once in the bert embeddings
        # and a seperate time directly
        text_type_embed = self.token_type_embeddings(segment_ids)
        return text_embedding + text_type_embed
