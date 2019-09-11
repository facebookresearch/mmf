# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.vqa.vizwiz import VizWizBuilder
from pythia.datasets.vqa.vqa2.ocr_dataset import VQA2OCRDataset


@Registry.register_builder("vqa2_ocr")
class TextVQABuilder(VizWizBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "VQA2_OCR"
        self.set_dataset_class(VQA2OCRDataset)
