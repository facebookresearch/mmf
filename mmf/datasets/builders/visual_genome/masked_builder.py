# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.visual_genome.builder import VisualGenomeBuilder
from mmf.datasets.builders.visual_genome.masked_dataset import MaskedVisualGenomeDataset


@registry.register_builder("masked_visual_genome")
class MaskedVisualGenomeBuilder(VisualGenomeBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_visual_genome"
        self.dataset_class = MaskedVisualGenomeDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/visual_genome/masked.yaml"
