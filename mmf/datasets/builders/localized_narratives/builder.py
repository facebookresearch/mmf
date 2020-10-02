# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.localized_narratives.dataset import (
    LocalizedNarrativesDataset,
)
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("localized_narratives")
class LocalizedNarrativesBuilder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="localized_narratives",
        dataset_class=LocalizedNarrativesDataset,
        *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(self):
        return "configs/datasets/localized_narratives/defaults.yaml"
