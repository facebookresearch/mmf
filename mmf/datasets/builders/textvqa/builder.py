# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import Registry
from mmf.datasets.builders.textvqa.dataset import TextVQADataset
from mmf.datasets.builders.vizwiz import VizWizBuilder


@Registry.register_builder("textvqa")
class TextVQABuilder(VizWizBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "textvqa"
        self.set_dataset_class(TextVQADataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/textvqa/defaults.yaml"
