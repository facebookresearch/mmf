# Copyright (c) Facebook, Inc. and its affiliates.

"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
 Adapted From Facebook Inc, Detectron2
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.import copy
 """
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.datasets.processors.processors import BaseProcessor
from mmf.utils.download import get_image_from_url
from PIL import Image


class ResizeShortestEdge:
    def __init__(self, short_edge_length: List[int], max_size: int = sys.maxsize):
        """
        Args:
            short_edge_length (list[min, max])
            max_size (int): maximum allowed longest edge length.
        """
        self.interp_method = "bilinear"
        self.max_size = max_size
        self.short_edge_length = short_edge_length

    def __call__(self, imgs: List[torch.Tensor]):
        img_augs = []
        for img in imgs:
            h, w = img.shape[:2]
            # later: provide list and randomly choose index for resize
            size = np.random.randint(
                self.short_edge_length[0], self.short_edge_length[1] + 1
            )
            if size == 0:
                return img
            scale = size * 1.0 / min(h, w)
            if h < w:
                newh, neww = size, scale * w
            else:
                newh, neww = scale * h, size
            if max(newh, neww) > self.max_size:
                scale = self.max_size * 1.0 / max(newh, neww)
                newh = newh * scale
                neww = neww * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)

            if img.dtype == np.uint8:
                pil_image = Image.fromarray(img)
                pil_image = pil_image.resize((neww, newh), Image.BILINEAR)
                img = np.asarray(pil_image)
            else:
                img = img.permute(2, 0, 1).unsqueeze(0)  # 3, 0, 1)  # hw(c) -> nchw
                img = F.interpolate(
                    img, (newh, neww), mode=self.interp_method, align_corners=False
                ).squeeze(0)
            img_augs.append(img)

        return img_augs


@registry.register_processor("frcnn_preprocess")
class FRCNNPreprocess(BaseProcessor):
    @dataclass
    class FRCNNPreprocessConfig:
        model: omegaconf.DictConfig = omegaconf.MISSING
        input: omegaconf.DictConfig = omegaconf.MISSING
        size_divisibility: int = 0
        pad_value: float = 0

    def __init__(self, config: FRCNNPreprocessConfig, *args, **kwargs):
        config_input = config.get("input", None)
        assert config_input is not None
        min_size_test = config_input.get("min_size_test", 800)
        max_size_test = config_input.get("max_size_test", 1333)
        self.aug = ResizeShortestEdge([min_size_test, min_size_test], max_size_test)
        self.input_format = config_input.get("format", "BGR")
        self.size_divisibility = config.get("size_divisibility", 0)
        self.pad_value = config.get("pad_value", 0)
        self.max_image_size = max_size_test
        config_model = config.get("model", None)
        assert config_model is not None
        self.device = config_model.get("device", "cpu")
        config_pixel_std = config_model.get("pixel_std", [1.0, 1.0, 1.0])
        self.pixel_std = (
            torch.tensor(config_pixel_std)
            .to(self.device)
            .view(len(config_pixel_std), 1, 1)
        )
        config_pixel_mean = config_model.get(
            "pixel_mean", [102.9801, 115.9465, 122.7717]
        )
        self.pixel_mean = (
            torch.tensor(config_pixel_mean)
            .to(self.device)
            .view(len(config_pixel_std), 1, 1)
        )
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

    def pad(self, images: List[torch.Tensor]):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        image_sizes = [im.shape[-2:] for im in images]
        images = [
            F.pad(
                im,
                [0, max_size[-1] - size[1], 0, max_size[-2] - size[0]],
                value=self.pad_value,
            )
            for size, im in zip(image_sizes, images)
        ]

        return torch.stack(images), torch.tensor(image_sizes)

    def __call__(self, images: torch.Tensor, single_image: bool = False):
        """
        Takes images of variable sizes, returns preprocessed
        version based on config sizing, etc.
        """
        with torch.no_grad():
            if not isinstance(images, list):
                images = [images]
            if single_image:
                assert len(images) == 1
            for i in range(len(images)):
                if isinstance(images[i], torch.Tensor):
                    images.insert(i, images.pop(i).to(self.device).float())
                elif not isinstance(images[i], torch.Tensor):
                    images.insert(
                        i,
                        torch.as_tensor(img_tensorize(images.pop(i)))
                        .to(self.device)
                        .float(),
                    )
            # resize smallest edge
            raw_sizes = torch.tensor([im.shape[:2] for im in images])
            images = self.aug(images)

            # flip rgb to bgr
            for idx in range(len(images)):
                images[idx] = torch.flip(images[idx], [0])
            # transpose images and convert to torch tensors
            # images = [torch.as_tensor(i.astype("float32"))
            # .permute(2, 0, 1).to(self.device) for i in images]
            # now normalize before pad to avoid useless arithmetic
            images = [self.normalizer(x) for x in images]
            # now pad them to do the following operations
            images, sizes = self.pad(images)
            # Normalize

            if self.size_divisibility > 0:
                raise NotImplementedError()
            # pad
            scales_yx = torch.true_divide(raw_sizes, sizes)
            if single_image:
                return images[0], sizes[0], scales_yx[0]
            else:
                return images, sizes, scales_yx


def img_tensorize(im: str):
    assert isinstance(im, str)
    if os.path.isfile(im):
        img = np.array(Image.open(im).convert("RGB"))
    else:
        img = get_image_from_url(im)
        assert img is not None, f"could not connect to: {im}"
    return img
