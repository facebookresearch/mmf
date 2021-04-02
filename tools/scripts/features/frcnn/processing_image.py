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
from typing import Tuple

import torch
import torch.nn.functional as F
from mmf.datasets.processors.frcnn_processor import ResizeShortestEdge, img_tensorize


class Preprocess:
    def __init__(self, cfg):
        self.aug = ResizeShortestEdge(
            [cfg.input.min_size_test, cfg.input.min_size_test], cfg.input.max_size_test
        )
        self.input_format = cfg.input.format
        self.size_divisibility = cfg.size_divisibility
        self.pad_value = cfg.PAD_VALUE
        self.max_image_size = cfg.input.max_size_test
        self.device = cfg.model.device
        self.pixel_std = (
            torch.tensor(cfg.model.pixel_std)
            .to(self.device)
            .view(len(cfg.model.pixel_std), 1, 1)
        )
        self.pixel_mean = (
            torch.tensor(cfg.model.pixel_mean)
            .to(self.device)
            .view(len(cfg.model.pixel_std), 1, 1)
        )
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

    def pad(self, images):
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

    def __call__(self, images, single_image=False):
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


def _scale_box(boxes, scale_yx):
    boxes[:, 0::2] *= scale_yx[:, 1]
    boxes[:, 1::2] *= scale_yx[:, 0]
    return boxes


def _clip_box(tensor, box_size: Tuple[int, int]):
    assert torch.isfinite(tensor).all(), "Box tensor contains infinite or NaN!"
    h, w = box_size
    tensor[:, 0].clamp_(min=0, max=w)
    tensor[:, 1].clamp_(min=0, max=h)
    tensor[:, 2].clamp_(min=0, max=w)
    tensor[:, 3].clamp_(min=0, max=h)
