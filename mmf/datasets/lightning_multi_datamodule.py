# Copyright (c) Facebook, Inc. and its affiliates.
import logging

from mmf.datasets.lightning_multi_dataset_loader import LightningMultiDataLoader
from mmf.datasets.multi_datamodule import MultiDataModule
from mmf.datasets.multi_dataset_loader import MultiDataLoader
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


class LightningMultiDataModule(MultiDataModule):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _build_multi_dataloader(self, dataset_type: "str" = "train") -> MultiDataLoader:
        loader_args = {}
        for key, datamodule in self.datamodules.items():
            loader_args[key] = getattr(datamodule, f"{dataset_type}_dataloader")()
            if not hasattr(loader_args[key], "dataset"):
                loader_args[key].dataset = getattr(
                    datamodule, f"{dataset_type}_dataset"
                )
        iteration_strategy = self._build_iteration_strategy(self.config, loader_args)
        loader = LightningMultiDataLoader(loader_args, iteration_strategy)
        return loader
