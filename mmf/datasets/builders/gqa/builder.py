# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.gqa.dataset import GQADataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("gqa")
class GQABuilder(MMFDatasetBuilder):
    def __init__(self, dataset_name="gqa", dataset_class=GQADataset, *args, **kwargs):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = GQADataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/gqa/defaults.yaml"

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
