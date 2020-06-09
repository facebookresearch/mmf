# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ChannelPool(nn.Module):
    """Average pooling in the channel dimension"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=1, keepdim=True)


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module in
    https://arxiv.org/pdf/1709.01507.pdf
    Args:
        dim: the original hidden dim.
        sqrate: the squeeze rate in hidden dim.
    Returns:
        New features map that channels are gated
        by sigmoid weights from SE module.
    """
    def __init__(self, dim, sqrate):
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

    def forward(self, x):
        x = x * self.se(x)

        return x * self.attn(x)


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight * self.running_var.rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)

            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return nn.functional.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=1e-5,
            )

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class Modulation(nn.Module):

    def __init__(self, num_features, num_cond_features, compressed=True):
        super(Modulation, self).__init__()
        self.linear = nn.Linear(num_cond_features, num_features)
        self.conv = nn.Conv2d(num_features, 256, kernel_size=1) \
            if compressed else nn.Conv2d(num_features, num_features, kernel_size=1)

    def forward(self, x, cond):
        cond = self.linear(cond).unsqueeze(2).unsqueeze(3)

        return self.conv(x * cond)


class Bottleneck(nn.Module):
    """
    Standard ResNet bottleneck with MoVie modulation in 
    https://arxiv.org/abs/2004.11883
    The code is inspired from
    https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnext101_32x8d
    """
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        cond_planes=None,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        stride_in_1x1=False,
        compressed=True,
        use_se=True,
    ):
        super().__init__()
        self.norm_layer = norm_layer
        self.cond_planes = cond_planes
        self.planes = planes
        self.inplanes = inplanes

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv1x1(inplanes, width, stride_1x1)
        self.conv2 = conv3x3(width, width, stride_3x3, groups, dilation)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se = None

        self.width = width
        self.compressed = compressed
        self.use_se = use_se

    def init_layers(self):
        if self.norm_layer is None:
            self.bn1 = FrozenBatchNorm2d(self.width)
            self.bn2 = FrozenBatchNorm2d(self.width)
            self.bn3 = FrozenBatchNorm2d(self.planes * self.expansion)
        else:
            self.bn1 = self.norm_layer(self.width)
            self.bn2 = self.norm_layer(self.width)
            self.bn3 = self.norm_layer(self.planes * self.expansion)

        if self.cond_planes:
            self.mod = Modulation(self.inplanes, self.cond_planes, compressed=self.compressed)
            self.se = SEModule(self.planes * self.expansion, 4) if self.use_se else None

    def forward(self, x, cond=None):
        identity = x

        if self.cond_planes and self.compressed:
            x = self.conv1(x) + self.mod(x, cond)
        elif self.cond_planes and not self.compressed:
            x += self.mod(x, cond)
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
