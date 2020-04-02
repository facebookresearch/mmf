# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import registry
from pythia.datasets.captioning.coco import MaskedCOCOBuilder


from .masked_dataset import MaskedCCDataset


@registry.register_builder("masked_cc")
class MaskedCCBuilder(MaskedCOCOBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_cc"
        self.set_dataset_class(MaskedCCDataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/conceptual_captions/masked.yaml"
