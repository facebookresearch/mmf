# Copyright (c) Facebook, Inc. and its affiliates.

from torch import Tensor


def transform_to_batch_sequence(tensor: Tensor) -> Tensor:
    if len(tensor.size()) == 2:
        return tensor
    else:
        assert len(tensor.size()) == 3
        return tensor.contiguous().view(-1, tensor.size(-1))


def transform_to_batch_sequence_dim(tensor: Tensor) -> Tensor:
    if len(tensor.size()) == 3:
        return tensor
    else:
        assert len(tensor.size()) == 4
        return tensor.contiguous().view(-1, tensor.size(-2), tensor.size(-1))
