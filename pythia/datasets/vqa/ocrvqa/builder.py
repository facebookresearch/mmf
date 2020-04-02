# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.vqa.ocrvqa.dataset import OCRVQADataset
from pythia.datasets.vqa.m4c_textvqa.builder import M4CTextVQABuilder


@Registry.register_builder("ocrvqa")
class OCRVQABuilder(M4CTextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "ocrvqa"
        self.set_dataset_class(OCRVQADataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/ocrvqa/defaults.yaml"
