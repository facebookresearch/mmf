# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from tests.test_utils import NumbersDataset


DATASET_LEN = 20


@registry.register_builder("always_one")
class AlwaysOneBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("always_one")

    def build(self, *args, **Kwargs):
        pass

    @classmethod
    def config_path(cls):
        return "configs/always_one.yaml"

    def load(self, config, dataset_type="train", *args, **kwargs):
        dataset = NumbersDataset(DATASET_LEN, data_item_key="input", always_one=True)
        dataset.dataset_name = self.dataset_name
        dataset.dataset_type = dataset_type
        return dataset
