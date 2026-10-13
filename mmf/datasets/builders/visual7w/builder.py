# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.visual7w.dataset import Visual7WDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("visual7w")
class Visual7WBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="visual7w", dataset_class=Visual7WDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)
        self.dataset_class = Visual7WDataset
        self._num_final_outputs = 4

    @classmethod
    def config_path(self):
        return "configs/datasets/visual7w/defaults.yaml"

    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor") and hasattr(
            self.dataset.text_processor, "get_vocab_size"
        ):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
        registry.register(
            self.dataset_name + "_num_final_outputs", self._num_final_outputs,
        )
