# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import Registry
from mmf.datasets.builders.m4c_textvqa.dataset import M4CTextVQADataset
from mmf.datasets.builders.textvqa.builder import TextVQABuilder


@Registry.register_builder("m4c_textvqa")
class M4CTextVQABuilder(TextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "m4c_textvqa"
        self.set_dataset_class(M4CTextVQADataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/m4c_textvqa/defaults.yaml"
