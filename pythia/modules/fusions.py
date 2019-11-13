# Copyright (c) Facebook, Inc. and its affiliates.
"""
The fusions module contains various Fusions techniques some based on BLOCK:
Bilinear Superdiagonal Fusion for VQA and VRD. For e.g. LinearSum, ConcatMLP
taken from https://github.com/Cadene/block.bootstrap.pytorch#fusions.

For implementing your own fusion technique, you need to follow these steps:

.. code::
    from pythia.common.registry import registry
    from pythia.modules.fusions import BaseFusion
    from torch import nn

    @regitery.register_fusion("custom")
    class CustomFusion(nn.Module):
        def __init__(self, params=None):
            super().__init__("Custom")

Then in your model's config you can specify ``fusion`` attribute as follows:

.. code::

   model_attributes:
      some_model:
          fusion:
             - type: block
             - params: {}
"""
import collections

import torch
import torch.nn as nn
from block import fusions

from pythia.common.registry import registry


@registry.register_fusion("block")
class Block(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(Block, self).__init__()
        self.module = fusions.Block(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

@registry.register_fusion("linear_sum")
class LinearSum(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(LinearSum, self).__init__()
        self.module = fusions.LinearSum(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@registry.register_fusion("concat_mlp")
class ConcatMLP(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(ConcatMLP, self).__init__()
        self.module = fusions.ConcatMLP(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@registry.register_fusion("mlb")
class MLB(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(MLB, self).__init__()
        self.module = fusions.MLB(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@registry.register_fusion("mutan")
class Mutan(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(Mutan, self).__init__()
        self.module = fusions.Mutan(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@registry.register_fusion("tucker")
class Tucker(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(Tucker, self).__init__()
        self.module = fusions.Tucker(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@registry.register_fusion("block_tucker")
class BlockTucker(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(BlockTucker, self).__init__()
        self.module = fusions.BlockTucker(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@registry.register_fusion("mfh")
class MFH(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(MFH, self).__init__()
        self.module = fusions.MFH(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@registry.register_fusion("mfb")
class MFB(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(MFB, self).__init__()
        self.module = fusions.MFB(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


@registry.register_fusion("mcb")
class MCB(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(MCB, self).__init__()
        self.module = fusions.MCB(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
