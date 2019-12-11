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
from block.fusions import (
    Block, LinearSum, ConcatMLP, MLB,
    Mutan, BlockTucker, Tucker, MFH,
    MFB, MCB
)

from pythia.common.registry import registry


registry.register_fusion("block")(Block)


registry.register_fusion("linear_sum")(LinearSum)


registry.register_fusion("concat_mlp")(ConcatMLP)


registry.register_fusion("mlb")(MLB)


registry.register_fusion("mutan")(Mutan)


registry.register_fusion("tucker")(Tucker)


registry.register_fusion("block_tucker")(BlockTucker)


registry.register_fusion("mfh")(MFH)


registry.register_fusion("mfb")(MFB)


registry.register_fusion("mcb")(MCB)
