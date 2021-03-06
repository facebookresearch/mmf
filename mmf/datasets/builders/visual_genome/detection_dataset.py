# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.datasets.builders.coco.detection_dataset import DetectionCOCODataset


class DetectionVisualGenomeDataset(DetectionCOCODataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "detection_visual_genome"
        self.dataset_name = name
