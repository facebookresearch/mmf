# Copyright (c) Facebook, Inc. and its affiliates.

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Tuple

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.encoders import IdentityEncoder
from mmf.modules.layers import AttnPool1d
from mmf.utils.build import (
    build_classifier_layer,
    build_image_encoder,
    build_text_encoder,
)
from mmf.utils.general import filter_grads
from mmf.utils.modeling import get_bert_configured_parameters
from mmf.utils.transform import transform_to_batch_sequence
from omegaconf import MISSING


class PositionEmbeddingSine(torch.nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: float = 10000,
        eps: float = 1e-06,
        normalize: bool = True,
        scale: bool = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.eps = eps
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        # input shape is B x 2048 x 7 x 7 or B x 2048 x 1 x 1
        x = tensor
        not_mask = torch.ones((x.shape[0], *x.shape[2:]), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class BaseAlign(BaseModel):
    @dataclass
    class Config(BaseModel.Config):
        # final layer mlp hidden size
        final_hidden_size: int = 512
        # whether to normalize the embedding
        norm_img_embeddings: bool = False
        norm_text_embeddings: bool = True
        direct_features_input: bool = False
        image_encoder: Any = MISSING
        text_encoder: Any = MISSING
        image_projection: Any = IdentityEncoder.Config()
        text_projection: Any = IdentityEncoder.Config()

    def __init__(self, config: Config):
        """Initialize the config which is the model configuration."""
        super().__init__(config)
        self.config = config

    def preprocess_text(self, sample_list) -> Tuple:
        raise NotImplementedError("Text processing not implemented")

    def preprocess_image(self, sample_list) -> Tuple:
        raise NotImplementedError("Image processing not implemented")

    def get_image_embeddings(self, sample_list) -> torch.Tensor:
        raise NotImplementedError("Image Encoder not implemented")

    def get_text_embedding(self, sample_list) -> torch.Tensor:
        raise NotImplementedError("Text Encoder not implemented")


@registry.register_model("cm_shared_transformer")
class CMSharedTransformer(BaseAlign):
    def __init__(self, config: BaseAlign.Config):
        """Initialize the config which is the model configuration."""
        super().__init__(config)
        self.config = config
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/alignment/defaults.yaml"

    def build(self):
        self._is_direct_features_input = self.config.direct_features_input
        # Encoders
        self.text_encoder = build_text_encoder(self.config.text_encoder)
        self.image_encoder = build_image_encoder(
            self.config.image_encoder, self._is_direct_features_input
        )

        # Projectors
        image_proj_config = deepcopy(self.config.image_projection)
        self.image_proj = build_classifier_layer(image_proj_config)

        text_proj_config = deepcopy(self.config.text_projection)
        self.text_proj = build_classifier_layer(text_proj_config)

        # Aggregators
        self.image_pool = AttnPool1d(self.config.final_hidden_size, 1)
        self.text_pool = AttnPool1d(self.config.final_hidden_size, 1)

        # Shared transformer
        transformer_layer = torch.nn.TransformerEncoderLayer(
            self.config.final_hidden_size, 4, 2048, dropout=0.1, activation="relu"
        )
        self.shared_transformer = torch.nn.TransformerEncoder(
            transformer_layer, num_layers=2
        )

        # Position embeddings - Image
        self.image_pos_emb = PositionEmbeddingSine(self.config.final_hidden_size // 2)

    def get_optimizer_parameters(self, config):
        base_lr = config.optimizer.params.lr
        bert_params = get_bert_configured_parameters(self.text_encoder, base_lr * 0.1)
        backbone_params = [
            {
                "params": filter_grads(self.image_encoder.parameters()),
                "lr": base_lr * 0.1,
            }
        ]
        rest_params = [
            {"params": filter_grads(self.image_proj.parameters()), "lr": base_lr},
            {"params": filter_grads(self.text_proj.parameters()), "lr": base_lr},
            {"params": filter_grads(self.image_pool.parameters()), "lr": base_lr},
            {"params": filter_grads(self.text_pool.parameters()), "lr": base_lr},
            {
                "params": filter_grads(self.shared_transformer.parameters()),
                "lr": base_lr,
            },
        ]
        training_parameters = bert_params + backbone_params + rest_params

        return training_parameters

    def preprocess_text(self, sample_list) -> Tuple:
        text = transform_to_batch_sequence(sample_list.input_ids)
        mask = transform_to_batch_sequence(sample_list.input_mask)
        segment = transform_to_batch_sequence(sample_list.segment_ids)

        return (text, mask, segment)

    def preprocess_image(self, sample_list) -> Tuple:
        if self._is_direct_features_input:
            return sample_list.image_feature_0.permute(0, 2, 1).unsqueeze(3)
            # return shape is B x 2048 x 1 x 1
        else:
            return sample_list.image

    def get_image_embeddings(self, sample_list) -> Tuple[torch.Tensor, torch.Tensor]:
        image_data = self.preprocess_image(sample_list)
        # image_data shape B x 3 x 224 x 224, B x 1 x 2048
        src = self.image_encoder(image_data)
        # src shape B x 49 x 2048, B x 1 x 2048
        if isinstance(src, dict):
            src = src[0]
        # Image embedding
        pos_src = src.permute(0, 2, 1)  # B x 2048 x 49,
        image_pos_embd = self.image_pos_emb(
            pos_src.reshape(
                (
                    pos_src.shape[0],
                    pos_src.shape[1],
                    int(math.sqrt(pos_src.shape[2])),
                    int(math.sqrt(pos_src.shape[2])),
                )
            )
        )
        image_pos_embd = image_pos_embd.flatten(2).permute(2, 0, 1)
        src_reshaped = src.flatten(2).permute(1, 0, 2)
        image_proj = self.image_proj(src_reshaped)
        # image_proj shape 49 x B x 512 1 x B x 512
        image_emb = image_proj + image_pos_embd

        # Shared transformer
        image_proj_sec = self.shared_transformer(image_emb)
        # Project to shared space
        image_proj_sec = image_proj_sec.permute(1, 0, 2)
        image_pool = self.image_pool(image_proj_sec, image_proj_sec).squeeze(1)
        if self.config.norm_img_embeddings:
            image_pool = torch.nn.functional.normalize(image_pool, 2, dim=1)

        return image_pool

    def get_text_embeddings(self, sample_list) -> Tuple[torch.Tensor, torch.Tensor]:
        text, mask, segment = self.preprocess_text(sample_list)

        text_enc = self.text_encoder(text, mask, segment)

        # Text embedding
        text_proj = self.text_proj(text_enc[0]).permute(1, 0, 2)
        text_ebd = text_proj

        # Shared transformer
        text_proj_sec = self.shared_transformer(
            text_ebd, src_key_padding_mask=mask.eq(0)
        )

        # Project to shared space
        text_proj_sec = text_proj_sec.permute(1, 0, 2)
        text_pool = self.text_pool(text_proj_sec, text_proj_sec, mask.eq(0)).squeeze(1)

        if self.config.norm_text_embeddings:
            text_pool = torch.nn.functional.normalize(text_pool, 2, dim=1)

        return text_pool

    def forward(self, sample_list):
        image_proj = self.get_image_embeddings(sample_list)
        text_proj = self.get_text_embeddings(sample_list)

        output = {
            "scores": image_proj,
            "targets": text_proj,
            "text_len": sample_list.input_mask.sum(-1).flatten(),
        }

        return output
