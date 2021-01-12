# Copyright (c) Facebook, Inc. and its affiliates.
import os
import pickle
import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torchvision
from mmf.common.registry import registry
from mmf.modules.embeddings import ProjectionEmbedding, TextEmbedding
from mmf.modules.hf_layers import BertModelJit
from mmf.modules.layers import Identity
from mmf.utils.build import build_image_encoder, build_text_encoder
from mmf.utils.download import download_pretrained_model
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_absolute_path
from omegaconf import MISSING, OmegaConf
from torch import nn
from transformers.configuration_auto import AutoConfig
from transformers.modeling_auto import AutoModel


try:
    from detectron2.modeling import ShapeSpec, build_resnet_backbone
except ImportError:
    pass


class Encoder(nn.Module):
    @dataclass
    class Config:
        name: str = MISSING

    @classmethod
    def from_params(cls, **kwargs):
        config = OmegaConf.structured(cls.Config(**kwargs))
        return cls(config)


class EncoderFactory(nn.Module):
    @dataclass
    class Config:
        type: str = MISSING
        params: Encoder.Config = MISSING


class ImageFeatureEncoderTypes(Enum):
    default = "default"
    identity = "identity"
    projection = "projection"
    frcnn_fc7 = "finetune_faster_rcnn_fpn_fc7"


class ImageFeatureEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        in_dim: int = MISSING


class ImageFeatureEncoderFactory(EncoderFactory):
    @dataclass
    class Config(EncoderFactory.Config):
        type: ImageFeatureEncoderTypes = MISSING
        params: ImageFeatureEncoder.Config = MISSING

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        encoder_type = config.type
        if isinstance(encoder_type, ImageFeatureEncoderTypes):
            encoder_type = encoder_type.value

        assert (
            "in_dim" in config.params
        ), "ImageFeatureEncoder require 'in_dim' param in config"
        params = config.params

        if encoder_type == "default" or encoder_type == "identity":
            self.module = Identity()
            self.module.in_dim = params.in_dim
            self.module.out_dim = params.in_dim
        elif encoder_type == "projection":
            if "module" not in params:
                params = deepcopy(params)
                params.module = "linear"
            self.module = ProjectionEmbedding(**params)
        elif encoder_type == "finetune_faster_rcnn_fpn_fc7":
            self.module = FinetuneFasterRcnnFpnFc7(params)
        else:
            raise NotImplementedError("Unknown Image Encoder: %s" % encoder_type)

        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@registry.register_encoder("finetune_faster_rcnn_fpn_fc7")
class FinetuneFasterRcnnFpnFc7(ImageFeatureEncoder):
    @dataclass
    class Config(ImageFeatureEncoder.Config):
        name: str = "finetune_faster_rcnn_fpn_fc7"
        in_dim: int = MISSING
        weights_file: str = "fc7_w.pkl"
        bias_file: str = "fc7_b.pkl"
        model_data_dir: str = MISSING

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        model_data_dir = get_absolute_path(config.model_data_dir)

        if not os.path.isabs(config.weights_file):
            weights_file = os.path.join(model_data_dir, config.weights_file)
        if not os.path.isabs(config.bias_file):
            bias_file = os.path.join(model_data_dir, config.bias_file)

        if not PathManager.exists(bias_file) or not PathManager.exists(weights_file):
            download_path = download_pretrained_model("detectron.vmb_weights")
            weights_file = get_absolute_path(os.path.join(download_path, "fc7_w.pkl"))
            bias_file = get_absolute_path(os.path.join(download_path, "fc7_b.pkl"))

        with PathManager.open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with PathManager.open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(config.in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        old_prefix = prefix + "module."
        for k in list(state_dict.keys()):
            if k.startswith(old_prefix):
                new_k = k.replace(old_prefix, prefix)
                state_dict[new_k] = state_dict.pop(k)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3


@registry.register_encoder("identity")
class IdentityEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "identity"
        in_dim: int = MISSING

    def __init__(self, config: Config):
        super().__init__()
        self.module = nn.Identity()
        self.in_dim = config.in_dim
        self.out_dim = config.in_dim

    def forward(self, x):
        return self.module(x)


class ImageEncoderTypes(Enum):
    default = "default"
    identity = "identity"
    resnet152 = "resnet152"
    detectron2_resnet = "detectron2_resnet"


class ImageEncoderFactory(EncoderFactory):
    @dataclass
    class Config(EncoderFactory.Config):
        type: ImageEncoderTypes = MISSING

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self._type = config.type

        if isinstance(self._type, ImageEncoderTypes):
            self._type = self._type.value

        params = config.params

        if self._type == "default" or self._type == "identity":
            self.module = nn.Identity()
            self.module.out_dim = params.in_dim
        elif self._type == "resnet152":
            self.module = ResNet152ImageEncoder(params)
        elif self._type == "detectron2_resnet":
            self.module = Detectron2ResnetImageEncoder(params)

        else:
            raise NotImplementedError("Unknown Image Encoder: %s" % self._type)

    @property
    def out_dim(self):
        return self.module.out_dim

    def forward(self, image):
        return self.module(image)


# Taken from facebookresearch/mmbt with some modifications
@registry.register_encoder("resnet152")
class ResNet152ImageEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "resnet152"
        pretrained: bool = True
        # "avg" or "adaptive"
        pool_type: str = "avg"
        num_output_features: int = 1

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config
        model = torchvision.models.resnet152(pretrained=config.get("pretrained", True))
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d if config.pool_type == "avg" else nn.AdaptiveMaxPool2d
        )

        # -1 will keep the original feature size
        if config.num_output_features == -1:
            self.pool = nn.Identity()
        elif config.num_output_features in [1, 2, 3, 5, 7]:
            self.pool = pool_func((config.num_output_features, 1))
        elif config.num_output_features == 4:
            self.pool = pool_func((2, 2))
        elif config.num_output_features == 6:
            self.pool = pool_func((3, 2))
        elif config.num_output_features == 8:
            self.pool = pool_func((4, 2))
        elif config.num_output_features == 9:
            self.pool = pool_func((3, 3))

        self.out_dim = 2048

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


@registry.register_encoder("detectron2_resnet")
class Detectron2ResnetImageEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "detectron2_resnet"
        pretrained: bool = True
        pretrained_path: str = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config
        pretrained = config.get("pretrained", False)
        pretrained_path = config.get("pretrained_path", None)

        self.resnet = build_resnet_backbone(config, ShapeSpec(channels=3))

        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                pretrained_path, progress=False
            )
            new_state_dict = OrderedDict()
            replace_layer = {"backbone.": ""}

            for key, value in state_dict["model"].items():
                new_key = re.sub(
                    r"(backbone\.)", lambda x: replace_layer[x.groups()[0]], key
                )
                new_state_dict[new_key] = value
            self.resnet.load_state_dict(new_state_dict, strict=False)

        self.out_dim = 2048

    def forward(self, x):
        x = self.resnet(x)
        return x["res5"]


class TextEncoderTypes(Enum):
    identity = "identity"
    transformer = "transformer"
    embedding = "embedding"


class TextEncoderFactory(EncoderFactory):
    @dataclass
    class Config(EncoderFactory.Config):
        # identity, transformer or embedding as of now
        type: TextEncoderTypes = MISSING
        params: Encoder.Config = MISSING

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self._type = config.type
        if isinstance(self._type, TextEncoderTypes):
            self._type = self._type.value

        if self._type == "identity":
            self.module = nn.Identity()
        elif self._type == "transformer":
            self._module = TransformerEncoder(config.params)
            self.module = self._module.module
        elif self._type == "embedding":
            self.module = TextEmbeddingEncoder(config.params)
        else:
            raise NotImplementedError(f"Unknown Text Encoder {self._type}")

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@registry.register_encoder("text_embedding")
class TextEmbeddingEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "text_embedding"
        operator: str = MISSING
        # Keeping this Any for now as this
        # needs a separate refactor PR.
        embedding_params: Any = MISSING

    def __init__(self, config: Config):
        super().__init__()
        self._operator = config.operator
        self._embedding_params = config.embedding_params

        self.module = TextEmbedding(
            self._embedding_params.type, **self._embedding_params.params
        )

    def forward(self, x):
        x = self.module(x)
        if self._operator == "sum":
            x = x.sum(dim=1)
        elif self._operator == "concat":
            x = torch.cat(x, dim=1)
        elif self._operator == "mul":
            x = torch.prod(x, dim=1)

        return x.squeeze()


@registry.register_encoder("transformer")
class TransformerEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "transformer"
        num_segments: int = 2
        bert_model_name: str = "bert-base-uncased"
        # Options below can be overridden to update the bert configuration used
        # to initialize the bert encoder. If some option is missing or
        # if you are using an encoder different then BERT, add extra parameters
        # by inheriting and extending this config
        # Those options will automatically override the options for your transformer
        # encoder's configuration. For e.g. vocab_size is missing here, just add
        # vocab_size: x to update the size of the vocabulary with which encoder is
        # initialized. If you update the default values, the transformer you
        # will get will be initialized from scratch.
        hidden_size: int = 768
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        output_attentions: bool = False
        output_hidden_states: bool = False

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config
        hf_params = {"config": self._build_encoder_config(config)}

        # For BERT models, initialize using Jit version
        if self.config.bert_model_name.startswith("bert-"):
            self.module = BertModelJit.from_pretrained(
                self.config.bert_model_name, **hf_params
            )
        else:
            self.module = AutoModel.from_pretrained(
                self.config.bert_model_name, **hf_params
            )
        self.embeddings = self.module.embeddings
        self.original_config = self.config
        self.config = self.module.config
        self._init_segment_embeddings()

    def _init_segment_embeddings(self):
        if self.original_config.get("num_segments", None):
            num_segments = self.original_config.num_segments
            if hasattr(self.embeddings, "token_type_embeddings"):
                new_embeds = nn.Embedding(num_segments, self.config.hidden_size)
                new_embeds.weight.data[:2].copy_(
                    self.embeddings.token_type_embeddings.weight
                )
                for idx in range(2, num_segments - 1):
                    new_embeds.weight.data[idx].copy_(
                        self.embeddings.token_type_embeddings.weight.data.mean(dim=0)
                    )
                self.embeddings.token_type_embeddings = new_embeds

    def _build_encoder_config(self, config: Config):
        return AutoConfig.from_pretrained(
            self.config.bert_model_name, **OmegaConf.to_container(self.config)
        )

    def forward(self, *args, **kwargs):
        # Only return pooled output
        return self.module(*args, **kwargs)[1]


class MultiModalEncoderBase(Encoder):
    __jit_unused_properties__ = ["encoder_config"]

    @dataclass
    class Config(Encoder.Config):
        # This actually is Union[ImageEncoderConfig, ImageFeatureEncoderConfig]
        modal_encoder: EncoderFactory.Config = ImageEncoderFactory.Config(
            type=ImageEncoderTypes.resnet152, params=ResNet152ImageEncoder.Config()
        )
        text_encoder: EncoderFactory.Config = TextEncoderFactory.Config(
            type=TextEncoderTypes.transformer, params=TransformerEncoder.Config()
        )
        direct_features_input: bool = False
        modal_hidden_size: int = 2048
        text_hidden_size: int = 768

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config

        self._modal_encoder_config = self.config.get("modal_encoder", None)

        self._is_direct_features_input = self.config.get("direct_features_input", False)

        self.build()
        self.modal_hidden_size = self.config.get("modal_hidden_size", None)
        self.text_hidden_size = self.config.get("text_hidden_size", None)

    def build(self):
        encoders = self._build_encoders(self.config)
        self.text_encoder, self.modal_encoder = encoders[0], encoders[1]

        self._encoder_config = None
        if self.text_encoder:
            self._encoder_config = self.text_encoder.config

    @property
    def encoder_config(self):
        return self._encoder_config

    def _build_encoders(self, config):
        text_encoder = None
        if config.get("text_encoder", None):
            text_encoder = build_text_encoder(config.text_encoder)

        modal_encoder = None
        if config.get("modal_encoder", None):
            modal_encoder = self._build_modal_encoder(config.modal_encoder)

        return (text_encoder, modal_encoder)

    def _build_modal_encoder(self, config):
        return build_image_encoder(
            config, direct_features=self._is_direct_features_input
        )
