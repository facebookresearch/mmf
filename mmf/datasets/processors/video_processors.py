# Copyright (c) Facebook, Inc. and its affiliates.

# TODO: Once internal torchvision transforms become stable either in torchvision
# or in pytorchvideo, move to use those transforms.
import random

import mmf.datasets.processors.functional as F
import torch
from mmf.common.registry import registry
from mmf.datasets.processors import BaseProcessor


@registry.register_processor("video_random_crop")
class VideoRandomCrop(BaseProcessor):
    def __init__(self, *args, size=None, **kwargs):
        super().__init__()
        if size is None:
            raise TypeError("Parameter 'size' is required")
        self.size = size

    @staticmethod
    def get_params(vid, output_size):
        """Get parameters for ``crop`` for a random crop."""
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        i, j, h, w = self.get_params(vid, self.size)
        return F.video_crop(vid, i, j, h, w)


@registry.register_processor("video_center_crop")
class VideoCenterCrop(BaseProcessor):
    def __init__(self, *args, size=None, **kwargs):
        super().__init__()
        if size is None:
            raise TypeError("Parameter 'size' is required")
        self.size = size

    def __call__(self, vid):
        return F.video_center_crop(vid, self.size)


@registry.register_processor("video_resize")
class VideoResize(BaseProcessor):
    def __init__(self, *args, size=None, **kwargs):
        if size is None:
            raise TypeError("Parameter 'size' is required")
        self.size = size

    def __call__(self, vid):
        return F.video_resize(vid, self.size)


@registry.register_processor("video_to_tensor")
class VideoToTensor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def __call__(self, vid):
        return F.video_to_normalized_float_tensor(vid)


@registry.register_processor("video_normalize")
class VideoNormalize(BaseProcessor):
    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__()
        if mean is None and std is None:
            raise TypeError("'mean' and 'std' params are required")
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return F.video_normalize(vid, self.mean, self.std)


@registry.register_processor("video_random_horizontal_flip")
class VideoRandomHorizontalFlip(BaseProcessor):
    def __init__(self, p=0.5, **kwargs):
        super().__init__()
        self.p = p

    def __call__(self, vid):
        if random.random() < self.p:
            return F.video_hflip(vid)
        return vid


@registry.register_processor("video_pad")
class Pad(BaseProcessor):
    def __init__(self, padding=None, fill=0, **kwargs):
        super().__init__()
        if padding is None:
            raise TypeError("Parameter 'padding' is required")
        self.padding = padding
        self.fill = fill

    def __call__(self, vid):
        return F.video_pad(vid, self.padding, self.fill)


@registry.register_processor("truncate_or_pad")
class TruncateOrPad(BaseProcessor):
    # truncate or add 0 until the desired output size
    def __init__(self, output_size=None, **kwargs):
        super().__init__()
        if output_size is None:
            raise TypeError("Parameter 'output_size' is required")
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        if sample.shape[1] >= self.output_size:
            return sample[0, : self.output_size]
        else:
            return torch.cat(
                (sample[0, :], torch.zeros(1, self.output_size - sample.shape[1])),
                axis=1,
            )
