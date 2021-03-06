# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.coco.detection_dataset import DetectionCOCODataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("detection_coco")
class DetectionCOCOBuilder(MMFDatasetBuilder):
    def __init__(self):
        super().__init__(
            dataset_name="detection_coco", dataset_class=DetectionCOCODataset
        )

    @classmethod
    def config_path(cls):
        return "configs/datasets/coco/detection.yaml"
