# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import Registry
from mmf.datasets.builders.stvqa.dataset import STVQADataset
from mmf.datasets.builders.textvqa.builder import TextVQABuilder


@Registry.register_builder("stvqa")
class STVQABuilder(TextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "stvqa"
        self.set_dataset_class(STVQADataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/stvqa/defaults.yaml"
