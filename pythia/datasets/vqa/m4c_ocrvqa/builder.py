# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.vqa.m4c_ocrvqa.dataset import M4COCRVQADataset
from pythia.datasets.vqa.m4c_textvqa.builder import M4CTextVQABuilder


@Registry.register_builder("m4c_ocrvqa")
class M4COCRVQABuilder(M4CTextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "m4c_ocrvqa"
        self.set_dataset_class(M4COCRVQADataset)
