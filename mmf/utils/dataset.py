# Copyright (c) Facebook, Inc. and its affiliates.
import warnings

import torch
from mmf.common.sample import Sample


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
        x = bbox.get("top_left_x", bbox["topLeftX"])
        y = bbox.get("top_left_y", bbox["topLeftY"])
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


def build_dataset_from_multiple_imdbs(config, dataset_cls, dataset_type):
    from mmf.datasets.concat_dataset import MMFConcatDataset

    if dataset_type not in config.imdb_files:
        warnings.warn(
            "Dataset type {} is not present in "
            "imdb_files of dataset config. Returning None. "
            "This dataset won't be used.".format(dataset_type)
        )
        return None

    imdb_files = config["imdb_files"][dataset_type]

    datasets = []

    for imdb_idx in range(len(imdb_files)):
        dataset = dataset_cls(dataset_type, imdb_idx, config)
        datasets.append(dataset)

    dataset = MMFConcatDataset(datasets)

    return dataset
