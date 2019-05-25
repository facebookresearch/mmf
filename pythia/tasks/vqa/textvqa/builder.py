# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.tasks.vqa.textvqa.dataset import TextVQADataset
from pythia.tasks.vqa.vizwiz import VizWizBuilder


@Registry.register_builder("textvqa")
class TextVQABuilder(VizWizBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "textvqa"
        self.set_dataset_class(TextVQADataset)
