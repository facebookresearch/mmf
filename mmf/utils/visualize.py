# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, List, Optional, Tuple

import torch
import torchvision


def visualize_images(
    images: List[Any], size: Optional[Tuple[int, int]] = (224, 224), *args, **kwargs
):
    """Visualize a set of images using torchvision's make grid function. Expects
    PIL images which it will convert to tensor and optionally resize them. If resize is
    not passed, it will only accept a list with single image

    Args:
        images (List[Any]): List of images to be visualized
        size (Optional[Tuple[int, int]], optional): Size to which Images can be resized.
            If not passed, the function will only accept list with single image.
            Defaults to (224, 224).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Visualization tools require matplotlib. "
            + "Install using pip install matplotlib."
        )
        raise

    transform_list = []

    assert (
        size is not None or len(images) == 1
    ), "If size is not passed, only one image can be visualized"

    if size is not None:
        transform_list.append(torchvision.transforms.Resize(size=size))

    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)

    img_tensors = torch.stack([transform(image) for image in images])
    grid = torchvision.utils.make_grid(img_tensors, *args, **kwargs)

    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0))
