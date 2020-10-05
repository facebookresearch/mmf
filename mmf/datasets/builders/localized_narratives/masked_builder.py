# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.localized_narratives.masked_dataset import (
    MaskedLocalizedNarrativesDataset,
)
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("masked_localized_narratives")
class MaskedLocalizedNarrativesBuilder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="masked_localized_narratives",
        dataset_class=MaskedLocalizedNarrativesDataset,
        *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/localized_narratives/masked.yaml"
