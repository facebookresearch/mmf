# Copyright (c) Facebook, Inc. and its affiliates.

from pythia.common.registry import registry
from pythia.datasets.captioning.coco import MaskedCOCOBuilder


from .masked_dataset import MaskedSBUDataset


@registry.register_builder("masked_sbu")
class MaskedSBUBuilder(MaskedCOCOBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_sbu"
        self.set_dataset_class(MaskedSBUDataset)
