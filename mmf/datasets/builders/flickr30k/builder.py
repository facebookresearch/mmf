# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.flickr30k.dataset import Flickr30kDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("flickr30k")
class Visual7WBuilder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="flickr30k",
        dataset_class=Flickr30kDataset,
        *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)
        self.dataset_class = Flickr30kDataset

    @classmethod
    def config_path(self):
        return "configs/datasets/flickr30k/defaults.yaml"

    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor") and hasattr(
            self.dataset.text_processor, "get_vocab_size"
        ):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
        registry.register(
            self.dataset_name + "_num_final_outputs", 4,
        )
