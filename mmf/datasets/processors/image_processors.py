# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import math
import random
import warnings

import torch
from mmf.common.registry import registry
from mmf.datasets.processors.processors import BaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms


@registry.register_processor("torchvision_transforms")
class TorchvisionTransforms(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        transform_params = config.transforms
        assert OmegaConf.is_dict(transform_params) or OmegaConf.is_list(
            transform_params
        )
        if OmegaConf.is_dict(transform_params):
            transform_params = [transform_params]

        transforms_list = []

        for param in transform_params:
            if OmegaConf.is_dict(param):
                # This will throw config error if missing
                transform_type = param.type
                transform_param = param.get("params", OmegaConf.create({}))
            else:
                assert isinstance(param, str), (
                    "Each transform should either be str or dict containing "
                    + "type and params"
                )
                transform_type = param
                transform_param = OmegaConf.create([])

            transform = getattr(transforms, transform_type, None)
            # If torchvision doesn't contain this, check our registry if we
            # implemented a custom transform as processor
            if transform is None:
                transform = registry.get_processor_class(transform_type)
            assert (
                transform is not None
            ), f"torchvision.transforms has no transform {transform_type}"

            # https://github.com/omry/omegaconf/issues/248
            transform_param = OmegaConf.to_container(transform_param)
            # If a dict, it will be passed as **kwargs, else a list is *args
            if isinstance(transform_param, collections.abc.Mapping):
                transform_object = transform(**transform_param)
            else:
                transform_object = transform(*transform_param)

            transforms_list.append(transform_object)

        self.transform = transforms.Compose(transforms_list)

    def __call__(self, x):
        # Support both dict and normal mode
        if isinstance(x, collections.abc.Mapping):
            x = x["image"]
            return {"image": self.transform(x)}
        else:
            return self.transform(x)


@registry.register_processor("GrayScaleTo3Channels")
class GrayScaleTo3Channels(BaseProcessor):
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, x):
        if isinstance(x, collections.abc.Mapping):
            x = x["image"]
            return {"image": self.transform(x)}
        else:
            return self.transform(x)

    def transform(self, x):
        assert isinstance(x, torch.Tensor)
        # Handle grayscale, tile 3 times
        if x.size(0) == 1:
            x = torch.cat([x] * 3, dim=0)
        return x


@registry.register_processor("ResizeShortest")
class ResizeShortest(BaseProcessor):
    def __init__(self, *args, **kwargs):
        min_size = kwargs["min_size"]
        max_size = kwargs["max_size"]
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(math.floor(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = transforms.functional.resize(image, size)

        return image


@registry.register_processor("NormalizeBGR255")
class NormalizeBGR255(BaseProcessor):
    def __init__(self, *args, **kwargs):
        self.mean = kwargs["mean"]
        self.std = kwargs["std"]
        self.to_bgr255 = kwargs["to_bgr255"]
        self.pad_size = kwargs["pad_size"]
        if self.pad_size > 0:
            warnings.warn(
                f"You are setting pad_size > 0, tensor will be padded to a fix size of"
                f"{self.pad_size}. "
                f"The image_mask will cover the pad_size of {self.pad_size} instead of"
                "the original size."
            )

    def __call__(self, image):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = transforms.functional.normalize(image, mean=self.mean, std=self.std)
        if self.pad_size > 0:
            assert (
                self.pad_size >= image.shape[1] and self.pad_size >= image.shape[2]
            ), f"image size: {image.shape}"
            padded_image = image.new_zeros(3, self.pad_size, self.pad_size)
            padded_image[:, : image.shape[1], : image.shape[2]] = image.clone()

            return padded_image
        return image
