# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.coco2017.masked_dataset import MaskedCoco2017Dataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("masked_coco2017")
class MaskedFlickr30kBuilder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="masked_coco2017",
        dataset_class=MaskedCoco2017Dataset,
        *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/coco2017/masked.yaml"
