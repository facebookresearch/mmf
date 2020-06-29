# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.flickr30k_retrieval.dataset import Flickr30kRetrievalDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("flickr30k_retrieval")
class Flickr30RetrievalBuilder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="flickr30k_retrieval",
        dataset_class=Flickr30kRetrievalDataset,
        *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)
        self.dataset_class = Flickr30kRetrievalDataset

    @classmethod
    def config_path(self):
        return "configs/datasets/flickr30k_retrieval/defaults.yaml"

    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor") and hasattr(
            self.dataset.text_processor, "get_vocab_size"
        ):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
        registry.register(
            self.dataset_name + "_num_final_outputs", config.num_final_outputs,
        )
