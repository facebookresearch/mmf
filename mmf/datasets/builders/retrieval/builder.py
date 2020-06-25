# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.retrieval.dataset import RetrievalDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("retrieval")
class RetrievalBuilder(MMFDatasetBuilder):
    def __init__(self, dataset_name="retrieval", dataset_class=RetrievalDataset, *args, **kwargs):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = RetrievalDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/retrieval/defaults.yaml"

    def load(self, config, dataset_type, *args, **kwargs):
        # Load the dataset using the RetrievalDataset class
        self.dataset = self.dataset_class(config, dataset_type)
        return self.dataset

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
