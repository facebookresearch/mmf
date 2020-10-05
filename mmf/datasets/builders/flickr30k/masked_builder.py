# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.flickr30k.masked_dataset import MaskedFlickr30kDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("masked_flickr30k")
class MaskedFlickr30kBuilder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="masked_flickr30k",
        dataset_class=MaskedFlickr30kDataset,
        *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/flickr30k/masked.yaml"
