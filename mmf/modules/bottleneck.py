# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from torchvision.models.resnet import conv1x1, conv3x3
from torchvision.ops.misc import FrozenBatchNorm2d


class ChannelPool(nn.Module):
    """Average pooling in the channel dimension"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True)


class SEModule(nn.Module):
    """Squeeze-and-Excitation module from https://arxiv.org/pdf/1709.01507.pdf

    Args:
        dim: the original hidden dim.
        sqrate: the squeeze rate in hidden dim.
    Returns:
        New features map that channels are gated
        by sigmoid weights from SE module.
    """

    def __init__(self, dim: int, sqrate: float):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim // sqrate, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // sqrate, dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.attn = nn.Sequential(
            ChannelPool(),
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.se(x)

        return x * self.attn(x)


class Modulation(nn.Module):
    def __init__(
        self, num_features: int, num_cond_features: int, compressed: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(num_cond_features, num_features)
        self.conv = (
            nn.Conv2d(num_features, 256, kernel_size=1)
            if compressed
            else nn.Conv2d(num_features, num_features, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = self.linear(cond).unsqueeze(2).unsqueeze(3)

        return self.conv(x * cond)


class MovieBottleneck(nn.Module):
    """
    Standard ResNet bottleneck with MoVie modulation in
    https://arxiv.org/abs/2004.11883
    The code is inspired from
    https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        cond_planes: int = None,
        stride: int = 1,
        downsample: Optional[Type[nn.Module]] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Type[nn.Module]] = None,
        stride_in_1x1: bool = False,
        compressed: bool = True,
        use_se: bool = True,
    ):
        super().__init__()
        if norm_layer is None:
            self.norm_layer = FrozenBatchNorm2d
        else:
            self.norm_layer = norm_layer
        self.cond_planes = cond_planes
        self.planes = planes
        self.inplanes = inplanes

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        self.width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv1x1(inplanes, self.width, stride_1x1)
        self.bn1 = self.norm_layer(self.width)
        self.conv2 = conv3x3(self.width, self.width, stride_3x3, groups, dilation)
        self.bn2 = self.norm_layer(self.width)
        self.conv3 = conv1x1(self.width, planes * self.expansion)
        self.bn3 = self.norm_layer(self.planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se = None

        self.compressed = compressed
        self.use_se = use_se

    def init_layers(self):
        if self.cond_planes:
            self.cond = Modulation(
                self.inplanes, self.cond_planes, compressed=self.compressed
            )
            self.se = SEModule(self.planes * self.expansion, 4) if self.use_se else None

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        identity = x

        if self.cond_planes and self.compressed:
            x = self.conv1(x) + self.cond(x, cond)
        elif self.cond_planes and not self.compressed:
            x += self.cond(x, cond)
            x = self.conv1(x)
        else:
            x = self.conv1(x)

        out = self.bn1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            shortcut = self.downsample(identity)
        else:
            shortcut = identity

        if self.se:
            out = self.se(out)

        out += shortcut
        out = self.relu(out)

        return out, cond
