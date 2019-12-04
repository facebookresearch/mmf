# Copyright (c) Facebook, Inc. and its affiliates.
"""
The fusions module contains various Fusions techniques some based on BLOCK:
Bilinear Superdiagonal Fusion for VQA and VRD. For e.g. LinearSum, ConcatMLP
taken from https://github.com/Cadene/block.bootstrap.pytorch#fusions.

For implementing your own fusion technique, you need to follow these steps:

.. code::
    from torch import nn
    from pythia.common.registry import registry
    from pythia.modules.fusions import Block
    from pythia.modules.fusions import LinearSum
    from pythia.modules.fusions import ConcatMLP
    from pythia.modules.fusions import MLB
    from pythia.modules.fusions import Mutan
    from pythia.modules.fusions import Tucker
    from pythia.modules.fusions import BlockTucker
    from pythia.modules.fusions import MFH
    from pythia.modules.fusions import MFB
    from pythia.modules.fusions import MCB

    @regitery.register_fusion("custom")
    class CustomFusion(nn.Module):
        def __init__(self, params=None):
            super().__init__("Custom")
"""
import collections

import torch
import torch.nn as nn
from block import fusions

from pythia.common.registry import registry


registry.register_fusion("block")(fusions.Block)
class Block(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(Block, self).__init__()
        self.module = fusions.Block(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

registry.register_fusion("linear_sum")(fusions.LinearSum)
class LinearSum(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(LinearSum, self).__init__()
        self.module = fusions.LinearSum(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


registry.register_fusion("concat_mlp")(fusions.ConcatMLP)
class ConcatMLP(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(ConcatMLP, self).__init__()
        self.module = fusions.ConcatMLP(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


registry.register_fusion("mlb")(fusions.MLB)
class MLB(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(MLB, self).__init__()
        self.module = fusions.MLB(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


registry.register_fusion("mutan")(fusions.Mutan)
class Mutan(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(Mutan, self).__init__()
        self.module = fusions.Mutan(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


registry.register_fusion("tucker")(fusions.Tucker)
class Tucker(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(Tucker, self).__init__()
        self.module = fusions.Tucker(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


registry.register_fusion("block_tucker")(fusions.BlockTucker)
class BlockTucker(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(BlockTucker, self).__init__()
        self.module = fusions.BlockTucker(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


registry.register_fusion("mfh")(fusions.MFH)
class MFH(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(MFH, self).__init__()
        self.module = fusions.MFH(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


registry.register_fusion("mfb")(fusions.MFB)
class MFB(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(MFB, self).__init__()
        self.module = fusions.MFB(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


registry.register_fusion("mcb")(fusions.MCB)
class MCB(nn.Module):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(MCB, self).__init__()
        self.module = fusions.MCB(input_dims, output_dims, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
