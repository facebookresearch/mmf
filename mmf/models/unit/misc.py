# Copyright (c) Facebook, Inc. and its affiliates.

# Mostly copy-pasted from
# https://github.com/facebookresearch/detr/blob/master/util/misc.py
from typing import List

import torch
from torch import Tensor


class NestedTensor:
    """
    A data class to hold images of different sizes in a batch.

    It contains `tensors` to hold padded images to the maximum size and `mask` to
    indicate the actual image region of each padded image
    """

    def __init__(self, tensors: Tensor, mask: Tensor):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list: List[Tensor]):
        """
        convert a list of images in CHW format in `tensor_list` to a NestedTensor by
        padding them to maximum image size.
        """
        if tensor_list[0].ndim == 3:
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        else:
            raise Exception("tensor_list must contain images in CHW format")
        return cls(tensor, mask)
