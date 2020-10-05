# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.builders.localized_narratives.masked_dataset import (
    MaskedLocalizedNarrativesDatasetMixin,
)
from mmf.datasets.mmf_dataset import MMFDataset


class MaskedFlickr30kDataset(MaskedLocalizedNarrativesDatasetMixin, MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "masked_flickr30k", config, dataset_type, index, *args, **kwargs
        )
