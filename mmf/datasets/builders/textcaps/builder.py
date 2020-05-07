# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import Registry
from mmf.datasets.builders.coco.dataset import COCODataset
from mmf.datasets.builders.textcaps.dataset import TextCapsDataset
from mmf.datasets.builders.textvqa.builder import TextVQABuilder


@Registry.register_builder("textcaps")
class TextCapsBuilder(TextVQABuilder):
    def __init__(
        self, dataset_name="textcaps", dataset_class=TextCapsDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/textcaps/defaults.yaml"

    def load(self, config, *args, **kwargs):
        annotation_style = config.get("annotation_style", self.dataset_name)
        if annotation_style == "coco":
            self.dataset_class = COCODataset

        dataset = super().load(config, *args, **kwargs)
        dataset.dataset_name = self.dataset_name
        return dataset
