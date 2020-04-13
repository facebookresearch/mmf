from mmf.common.registry import registry
from mmf.datasets.builders.coco.builder import COCOBuilder

from .masked_dataset import MaskedCOCODataset


@registry.register_builder("masked_coco")
class MaskedCOCOBuilder(COCOBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_coco"
        self.set_dataset_class(MaskedCOCODataset)

    def update_registry_for_model(self, config):
        registry.register(
            self.dataset_name + "_text_vocab_size",
            self.dataset.masked_token_processor.get_vocab_size(),
        )

    @classmethod
    def config_path(cls):
        return "configs/datasets/coco/masked.yaml"
