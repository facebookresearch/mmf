from mmf.common.registry import registry
from mmf.datasets.builders.charades.dataset import CharadesDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("charades")
class CharadesBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="charades", dataset_class=CharadesDataset, *args, **kwargs
    ):
        super().__init__(dataset_name)
        self.dataset_class = CharadesDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/charades/defaults.yaml"
