# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.vqa.m4c_stvqa.dataset import M4CSTVQADataset
from pythia.datasets.vqa.m4c_textvqa.builder import M4CTextVQABuilder


@Registry.register_builder("m4c_stvqa")
class M4CSTVQABuilder(M4CTextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "m4c_stvqa"
        self.set_dataset_class(M4CSTVQADataset)
