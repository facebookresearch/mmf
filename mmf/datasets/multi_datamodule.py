# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import warnings
from typing import Dict, List, Optional

import pytorch_lightning as pl
from mmf.common.sample import SampleList
from mmf.common.test_reporter import TestReporter
from mmf.datasets.iteration_strategies import IterationStrategy
from mmf.datasets.multi_dataset_loader import MultiDataLoader
from mmf.utils.build import (
    build_iteration_strategy,
    build_multiple_datamodules,
    build_test_reporter,
)
from mmf.utils.dataset import dataset_list_from_config
from mmf.utils.general import get_batch_size
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class MultiDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.batch_size = get_batch_size()

        self.dataset_list: List[str] = dataset_list_from_config(self.config)
        self.datamodules: Dict[
            str, pl.LightningDataModule
        ] = build_multiple_datamodules(self.dataset_list, self.config.dataset_config)
        self.train_loader: Optional[MultiDataLoader] = None
        self.val_loader: Optional[MultiDataLoader] = None
        self.test_loader: Optional[MultiDataLoader] = None

    def train_dataloader(self) -> MultiDataLoader:
        self.train_loader = self._build_multi_dataloader("train")
        return self.train_loader

    def val_dataloader(self) -> MultiDataLoader:
        self.val_loader = self._build_multi_dataloader("val")
        return self.val_loader

    def test_dataloader(self) -> MultiDataLoader:
        self.test_loader = self._build_multi_dataloader("test")
        return self.test_loader

    def _build_iteration_strategy(
        self, config: DictConfig, dataloaders: Dict[str, DataLoader]
    ) -> IterationStrategy:
        disabled = OmegaConf.create({"enabled": False})

        if len(self.dataset_list) == 1:
            logger.info("Multitasking disabled by default for single dataset training")
            multitasking_config = disabled
        elif "multitasking" in self.config:
            multitasking_config = self.config.multitasking
        else:
            warnings.warn(
                "'multitasking' config not defined. Disabling any form of multitasking"
            )
            multitasking_config = disabled

        return build_iteration_strategy(multitasking_config, dataloaders)

    def _build_multi_dataloader(self, dataset_type: "str" = "train") -> MultiDataLoader:
        loader_args = {}
        for key, datamodule in self.datamodules.items():
            loader_args[key] = getattr(datamodule, f"{dataset_type}_dataloader")()
            if not hasattr(loader_args[key], "dataset"):
                loader_args[key].dataset = getattr(
                    datamodule, f"{dataset_type}_dataset"
                )
        iteration_strategy = self._build_iteration_strategy(self.config, loader_args)
        loader = MultiDataLoader(loader_args, iteration_strategy)
        return loader

    def teardown(self, *args, **kwargs):
        for _, datamodule in self.datamodules.items():
            if hasattr(datamodule, "teardown"):
                datamodule.teardown()

    ############################################################
    ######## Functions below are required for MMFTrainer #######
    ########      and not used by the PL Trainer         #######
    ############################################################
    def get_test_reporter(self, dataset_type: str) -> TestReporter:
        test_reporter_config = self._get_test_reporter_config()
        return build_test_reporter(self.datamodules, test_reporter_config, dataset_type)

    def _get_test_reporter_config(self):
        from mmf.utils.configuration import get_global_config

        return get_global_config("evaluation.reporter")

    def prepare_batch(self, batch, *args, **kwargs):
        batch = SampleList(batch)
        loader = self.get_loader(batch.dataset_type)
        return loader.prepare_batch(batch)

    def get_loader(self, dataset_type: str) -> MultiDataLoader:
        return getattr(self, f"{dataset_type}_loader")

    def seed_sampler(self, dataset_type: "str", seed: int):
        loader = self.get_loader(dataset_type)
        loader.seed_sampler(seed)
