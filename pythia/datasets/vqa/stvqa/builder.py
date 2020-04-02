# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.vqa.stvqa.dataset import STVQADataset
from pythia.datasets.vqa.m4c_textvqa.builder import M4CTextVQABuilder


@Registry.register_builder("stvqa")
class STVQABuilder(M4CTextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "stvqa"
        self.set_dataset_class(STVQADataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/stvqa/defaults.yaml"
