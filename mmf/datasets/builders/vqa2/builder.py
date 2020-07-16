# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from mmf.common.registry import registry
from mmf.datasets.builders.vqa2.dataset import VQA2Dataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("vqa2")
class VQA2Builder(MMFDatasetBuilder):
    def __init__(self, dataset_name="vqa2", dataset_class=VQA2Dataset, *args, **kwargs):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = VQA2Dataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/vqa2/defaults.yaml"

    def load(self, *args, **kwargs):
        dataset = super().load(*args, **kwargs)
        if dataset is not None and hasattr(dataset, "try_fast_read"):
            dataset.try_fast_read()

        return dataset

    # TODO: Deprecate this method and move configuration updates directly to processors
    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor"):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
        if hasattr(self.dataset, "answer_processor"):
            registry.register(
                self.dataset_name + "_num_final_outputs",
                self.dataset.answer_processor.get_vocab_size(),
            )


@registry.register_builder("vqa2_train_val")
class VQA2TrainValBuilder(VQA2Builder):
    def __init__(self, dataset_name="vqa2_train_val"):
        super().__init__(dataset_name)

    @classmethod
    def config_path(self):
        return "configs/datasets/vqa2/train_val.yaml"
