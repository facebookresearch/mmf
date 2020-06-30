# Copyright (c) Facebook, Inc. and its affiliates.
import os
import pickle
from typing import Dict, Union

import torch
import torchvision
from omegaconf import OmegaConf
from torch import nn
from transformers.configuration_auto import AutoConfig
from transformers.modeling_auto import AutoModel

from mmf.modules.embeddings import ProjectionEmbedding, TextEmbedding
from mmf.modules.layers import Identity
from mmf.utils.build import build_image_encoder, build_text_encoder
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.download import download_pretrained_model
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_absolute_path


class ImageFeatureEncoder(nn.Module):
    def __init__(self, encoder_type, in_dim, **kwargs):
        super().__init__()

        if encoder_type == "default":
            self.module = Identity()
            self.module.in_dim = in_dim
            self.module.out_dim = in_dim
        elif encoder_type == "projection":
            module_type = kwargs.pop("module", "linear")
            self.module = ProjectionEmbedding(module_type, in_dim, **kwargs)
        elif encoder_type == "finetune_faster_rcnn_fpn_fc7":
            self.module = FinetuneFasterRcnnFpnFc7(in_dim, **kwargs)
        else:
            raise NotImplementedError("Unknown Image Encoder: %s" % encoder_type)

        self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class FinetuneFasterRcnnFpnFc7(nn.Module):
    def __init__(
        self, in_dim, weights_file, bias_file, model_data_dir, *args, **kwargs
    ):
        super().__init__()
        model_data_dir = get_absolute_path(model_data_dir)

        if not os.path.isabs(weights_file):
            weights_file = os.path.join(model_data_dir, weights_file)
        if not os.path.isabs(bias_file):
            bias_file = os.path.join(model_data_dir, bias_file)

        if not PathManager.exists(bias_file) or not PathManager.exists(weights_file):
            download_path = download_pretrained_model("detectron.vmb_weights")
            weights_file = get_absolute_path(os.path.join(download_path, "fc7_w.pkl"))
            bias_file = get_absolute_path(os.path.join(download_path, "fc7_b.pkl"))

        with PathManager.open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with PathManager.open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3


class ImageEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self._type = config.type
        params = config.params

        if self._type == "default":
            self.module = nn.Identity()
            self.module.out_dim = params.in_dim
        elif self._type == "resnet152":
            self.module = ResNet152ImageEncoder(params)
        elif self._type.startswith("torchvision"):
            # Get model name, e.g. "torchvision::resnet50" -> "resnet50"
            torchvision_model_name = self._type.split("::")[1]

            self.module = TorchvisionImageEncoder(
                torchvision_model_name,
                pretrained=config.get("pretrained", False),
                frozen=config.get("frozen", False),
            )
        else:
            raise NotImplementedError("Unknown Image Encoder: %s" % self._type)

    @property
    def out_dim(self):
        return self.module.out_dim

    def forward(self, image):
        return self.module(image)


# Taken from facebookresearch/mmbt with some modifications
class ResNet152ImageEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
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


class TorchvisionImageEncoder(nn.Module):
    r"""
    An image encoder from `Torchvision model zoo
    <https://pytorch.org/docs/stable/torchvision/models.html>`_. Any model can
    be specified using corresponding method name from the model zoo.

    Parameters
    ----------
    name: str, optional (default = "resnet50")
        Name of the model from Torchvision model zoo.
    pretrained: bool, optional (default = False)
        Whether to load ImageNet pretrained weights from Torchvision.
    frozen: float, optional (default = False)
        Whether to keep all weights frozen during training.
    """

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = False,
        frozen: bool = False,
    ):
        super().__init__()

        # Instantiate a model from torchvision.
        self.cnn = getattr(torchvision.models, name)(
            pretrained, zero_init_residual=True,
        )
        # Do nothing after the final residual stage.
        self.cnn.fc = nn.Identity()

        # Freeze all weights if specified.
        if frozen:
            for param in self.cnn.parameters():
                param.requires_grad = False
            self.cnn.eval()

        # Keep a list of intermediate residual stage names.
        # NOTE: only work for ResNet-like models (do not work for VGG).
        self._stage_names = [f"layer{i}" for i in range(1, 5)]

        # Get visual feature size (channel dimension) by passing a 
        # random input tensor.
        with torch.no_grad():
            # Global average pooled features.
            output_vector = self.cnn(torch.randn(1, 3, 224, 224))
            self.visual_feature_size = output_vector.size(1)

    @property
    def out_dim(self):
        return self.visual_feature_size

    def forward(
        self, image: torch.Tensor, return_intermediate_outputs: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Compute visual features for a batch of input images.

        Parameters
        ----------
        image: torch.Tensor
            Batch of input images. A tensor of shape
            ``(batch_size, 3, image_height, image_width)``.
        return_intermediate_outputs: bool, optional (default = False)
            Whether to return feaures extracted from all intermediate stages
            or just the last one. This should only be set ``True`` when using
            a ResNet-like model.
 
        Returns
        -------
        Union[torch.Tensor, Dict[str, torch.Tensor]]
            - If ``return_intermediate_outputs = False``, this will be a tensor
              of shape ``(batch_size, grid_height * grid_width, out_dim)``.
              For example it will be ``(batch_size, 49, 2048)`` for ResNet-50.
            - If ``return_intermediate_outputs = True``, this will be a dict
              with keys ``{"layer1", "layer2", "layer3", "layer4", "avgpool"}``
              containing features from all intermediate layers and global
              average pooling layer.
        """

        # Iterate through the modules in sequence and collect feature
        # vectors for last layers in each stage.
        intermediate_outputs: Dict[str, torch.Tensor] = {}
        for idx, (name, layer) in enumerate(self.cnn.named_children()):
            # shape: (batch_size, feature_dim, feature_height, feature_width)
            out = layer(image) if idx == 0 else layer(out)

            if name in self._stage_names:
                # Bring channels to last dim and collapse rest of the dims,
                # except batch dim.
                # shape: (batch_size, grid_size, feature_dim)
                out = torch.flatten(out, start_dim=2)
                out = out.transpose(1, 2).contiguous()
                intermediate_outputs[name] = out

        # Add global average pooled spatial grid features.
        intermediate_outputs["avgpool"] = torch.mean(
            intermediate_outputs["layer4"], dim=1
        )
        if return_intermediate_outputs:
            return intermediate_outputs
        else:
            # shape: (batch_size, grid_size, out_dim)
            return intermediate_outputs["layer4"]


class TextEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self._type = config.type

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


class TextEmbeddingEncoder(nn.Module):
    def __init__(self, config):
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


class TransformerEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.module = AutoModel.from_pretrained(
            self.config.bert_model_name,
            config=self._build_encoder_config(config),
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
        )
        self.embeddings = self.module.embeddings
        self.config = self.module.config

    def _build_encoder_config(self, config):
        return AutoConfig.from_pretrained(
            self.config.bert_model_name, **OmegaConf.to_container(self.config)
        )

    def forward(self, *args, **kwargs):
        # Only return pooled output
        return self.module(*args, **kwargs)[1]


class MultiModalEncoderBase(nn.Module):
    def __init__(self, config, *args, **kwargs):
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
