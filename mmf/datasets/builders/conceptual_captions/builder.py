# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.coco import COCOBuilder

from .dataset import ConceptualCaptionsDataset


@registry.register_builder("conceptual_captions")
class ConceptualCaptionsBuilder(COCOBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "conceptual_captions"
        self.set_dataset_class(ConceptualCaptionsDataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/conceptual_captions/defaults.yaml"
