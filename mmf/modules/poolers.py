# Copyright (c) Facebook, Inc. and its affiliates.
"""
This module contains implementations for various pooling methods from
transformer encoder layers
.. code::

   from mmf.common.registry import registry
   from torch import nn


   @registry.register_pooler("custom")
   class CustomPool(nn.Module):
       ...
"""
from typing import List

import torch
import torch.nn as nn
from mmf.common.registry import registry


@registry.register_pooler("average_concat_last_k")
class AverageConcatLastN(nn.Module):
    def __init__(self, k=4, tol=0.000001):
        super().__init__()
        self.num_layers = k
        self.tol = tol

    def forward(self, encoded_layers: List[torch.Tensor], pad_mask: torch.Tensor):
        assert self.num_layers <= len(
            encoded_layers
        ), "k should be less than the number of encoder layers"
        encoder_avg = torch.cat(encoded_layers[-self.num_layers :], 2)

        pad_mask = pad_mask.unsqueeze(2)
        encoder_avg = encoder_avg * pad_mask.float()
        pooled_output = torch.sum(encoder_avg, 1) / (
            torch.sum(pad_mask, 1).float() + self.tol
        )
        return pooled_output


@registry.register_pooler("average_k_from_last")
class AverageKFromLast(nn.Module):
    def __init__(self, k=2, tol=0.000001):
        super().__init__()
        self.k = k
        self.tol = tol

    def forward(self, encoded_layers: List[torch.Tensor], pad_mask: torch.Tensor):
        assert self.k <= len(
            encoded_layers
        ), "k should be less than the number of encoder layers"
        encoder_avg = encoded_layers[-self.k]
        pad_mask = pad_mask.unsqueeze(2)
        encoder_avg = encoder_avg * pad_mask.float()
        pooled_output = torch.sum(encoder_avg, 1) / (
            torch.sum(pad_mask, 1).float() + self.tol
        )
        return pooled_output


@registry.register_pooler("average_sum_last_k")
class AverageSumLastK(nn.Module):
    def __init__(self, k=4, tol=0.000001):
        super().__init__()
        self.k = k
        self.tol = tol

    def forward(self, encoded_layers: List[torch.Tensor], pad_mask: torch.Tensor):
        assert self.k <= len(
            encoded_layers
        ), "k should be less than the number of encoder layers"
        encoder_avg = torch.stack(encoded_layers[-self.k :]).sum(0)
        pad_mask = pad_mask.unsqueeze(2)
        encoder_avg = encoder_avg * pad_mask.float()
        pooled_output = torch.sum(encoder_avg, 1) / (
            torch.sum(pad_mask, 1).float() + self.tol
        )
        return pooled_output
