# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import Registry
from mmf.datasets.builders.m4c_textvqa.builder import M4CTextVQABuilder
from mmf.datasets.builders.stvqa.dataset import STVQADataset


@Registry.register_builder("stvqa")
class STVQABuilder(M4CTextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "stvqa"
        self.set_dataset_class(STVQADataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/stvqa/defaults.yaml"
