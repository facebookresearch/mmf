# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.visual_genome.detection_dataset import (
    DetectionVisualGenomeDataset,
)
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("detection_visual_genome")
class DetectionVisualGenomeBuilder(MMFDatasetBuilder):
    def __init__(self):
        super().__init__(
            dataset_name="detection_visual_genome",
            dataset_class=DetectionVisualGenomeDataset,
        )

    @classmethod
    def config_path(cls):
        return "configs/datasets/visual_genome/detection.yaml"
