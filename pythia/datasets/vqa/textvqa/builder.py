# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.vqa.textvqa.dataset import TextVQADataset
from pythia.datasets.vqa.vizwiz import VizWizBuilder


@Registry.register_builder("textvqa")
class TextVQABuilder(VizWizBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "textvqa"
        self.set_dataset_class(TextVQADataset)
