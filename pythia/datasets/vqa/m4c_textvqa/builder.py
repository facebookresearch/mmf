# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.vqa.m4c_textvqa.dataset import M4CTextVQADataset
from pythia.datasets.vqa.textvqa.builder import TextVQABuilder


@Registry.register_builder("m4c_textvqa")
class M4CTextVQABuilder(TextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "m4c_textvqa"
        self.set_dataset_class(M4CTextVQADataset)
