# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.coco.dataset import COCODataset
from mmf.datasets.builders.textcaps.dataset import TextCapsDataset
from mmf.datasets.builders.vqa2 import VQA2Builder


@registry.register_builder("coco")
class COCOBuilder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "coco"
        self.set_dataset_class(COCODataset)

    # TODO: Deprecate this method and move configuration updates directly to processors
    def update_registry_for_model(self, config):
        registry.register(
            self.dataset_name + "_text_vocab_size",
            self.dataset.text_processor.get_vocab_size(),
        )

        if hasattr(self.dataset, "answer_processor"):
            registry.register(
                self.dataset_name + "_num_final_outputs",
                self.dataset.answer_processor.get_vocab_size(),
            )

            registry.register(
                self.dataset_name + "_answer_processor", self.dataset.answer_processor
            )

    @classmethod
    def config_path(cls):
        return "configs/datasets/coco/defaults.yaml"

    def load(self, config, *args, **kwargs):
        annotation_style = config.get("annotation_style", self.dataset_name)
        if annotation_style == "textcaps":
            self.dataset_class = TextCapsDataset

        dataset = super().load(config, *args, **kwargs)
        dataset.dataset_name = self.dataset_name
        return dataset
