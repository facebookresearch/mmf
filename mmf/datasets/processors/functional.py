# Copyright (c) Facebook, Inc. and its affiliates.

from typing import List, Tuple, Union

import torch


# Functional file similar to torch.nn.functional
def video_crop(vid: torch.tensor, i: int, j: int, h: int, w: int) -> torch.Tensor:
    return vid[..., i : (i + h), j : (j + w)]


def video_center_crop(vid: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return video_crop(vid, i, j, th, tw)


def video_hflip(vid: torch.Tensor) -> torch.Tensor:
    return vid.flip(dims=(-1,))


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def video_resize(
    vid: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    interpolation: str = "bilinear",
) -> torch.Tensor:
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False
    )


def video_pad(
    vid: torch.Tensor,
    padding: List[int],
    fill: float = 0,
    padding_mode: str = "constant",
) -> torch.Tensor:
    # NOTE: don't want to pad on temporal dimension, so let as non-batch
    # (4d) before padding. This works as expected
    return torch.nn.functional.pad(vid, padding, value=fill, mode=padding_mode)


def video_to_normalized_float_tensor(vid: torch.Tensor) -> torch.Tensor:
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


def video_normalize(
    vid: torch.Tensor, mean: List[float], std: List[float]
) -> torch.Tensor:
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std
