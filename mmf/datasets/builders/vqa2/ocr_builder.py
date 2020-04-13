# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import Registry
from mmf.datasets.builders.vizwiz import VizWizBuilder
from mmf.datasets.builders.vqa2.ocr_dataset import VQA2OCRDataset


@Registry.register_builder("vqa2_ocr")
class TextVQABuilder(VizWizBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "VQA2_OCR"
        self.set_dataset_class(VQA2OCRDataset)

    @classmethod
    def config_path(self):
        return None
