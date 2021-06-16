# Copyright (c) Facebook, Inc. and its affiliates.

# Mostly copy-pasted from
# https://github.com/facebookresearch/detr/blob/master/datasets/transforms.py
import random
from typing import List, Optional, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from mmf.common.registry import registry
from mmf.datasets.processors.processors import BaseProcessor
from mmf.utils.box_ops import box_xyxy_to_cxcywh
from torch import Tensor


def crop(image: Tensor, target: dict, region: List[int]):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "attributes" in target:
        fields.append("attributes")

    # remove elements for which the boxes have zero area
    if "boxes" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        cropped_boxes = target["boxes"].reshape(-1, 2, 2)
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image: Tensor, target: dict):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    return flipped_image, target


def get_size_with_aspect_ratio(
    image_size: List[int], size: int, max_size: Optional[int] = None
):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)


def get_size(
    image_size: List[int], size: Union[int, List[int]], max_size: Optional[int] = None
):
    if isinstance(size, (list, tuple)):
        return size[::-1]
    else:
        return get_size_with_aspect_ratio(image_size, size, max_size)


def resize(
    image: Tensor,
    target: dict,
    size: Union[int, List[int]],
    max_size: Optional[int] = None,
):
    # size can be min_size (scalar) or (w, h) tuple

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


def pad(image: Tensor, target: dict, padding: List[int]):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    return padded_image, target


@registry.register_processor("detection_random_size_crop")
class RandomSizeCrop(BaseProcessor):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: Tensor, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


@registry.register_processor("detection_random_horizontal_flip")
class RandomHorizontalFlip(BaseProcessor):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: Tensor, target: dict):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


@registry.register_processor("detection_random_resize")
class RandomResize(BaseProcessor):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img: Tensor, target: Optional[dict] = None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


@registry.register_processor("detection_random_select")
class RandomSelect(BaseProcessor):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img: Tensor, target: dict):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


@registry.register_processor("detection_to_tensor")
class ToTensor(BaseProcessor):
    def __init__(self):
        pass

    def __call__(self, img: Tensor, target: dict):
        return F.to_tensor(img), target


@registry.register_processor("detection_normalize")
class Normalize(BaseProcessor):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image: Tensor, target: Optional[dict] = None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


@registry.register_processor("detection_compose")
class Compose(BaseProcessor):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Tensor, target: dict):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
