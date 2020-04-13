# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import Registry
from mmf.datasets.builders.m4c_textvqa.builder import M4CTextVQABuilder
from mmf.datasets.builders.ocrvqa.dataset import OCRVQADataset


@Registry.register_builder("ocrvqa")
class OCRVQABuilder(M4CTextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "ocrvqa"
        self.set_dataset_class(OCRVQADataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/ocrvqa/defaults.yaml"
