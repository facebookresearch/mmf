# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from pythia.common.sample import Sample


def build_bbox_tensors(infos, max_length):
    num_bbox = min(max_length, len(infos))

    # After num_bbox, everything else should be zero
    coord_tensor = torch.zeros((max_length, 4), dtype=torch.float)
    width_tensor = torch.zeros(max_length, dtype=torch.float)
    height_tensor = torch.zeros(max_length, dtype=torch.float)
    bbox_types = ["xyxy"] * max_length

    infos = infos[:num_bbox]
    sample = Sample()

    for idx, info in enumerate(infos):
        bbox = info["bounding_box"]
        x = bbox["top_left_x"]
        y = bbox["top_left_y"]
        width = bbox["width"]
        height = bbox["height"]

        coord_tensor[idx][0] = x
        coord_tensor[idx][1] = y
        coord_tensor[idx][2] = x + width
        coord_tensor[idx][3] = y + height

        width_tensor[idx] = width
        height_tensor[idx] = height
    sample.coordinates = coord_tensor
    sample.width = width_tensor
    sample.height = height_tensor
    sample.bbox_types = bbox_types

    return sample
