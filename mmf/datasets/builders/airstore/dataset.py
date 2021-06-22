# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
from io import BytesIO
from typing import Any, Iterable

import torch
from iopath.common.file_io import PathManager
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.general import get_batch_size
from PIL import Image, ImageFile


logger = logging.getLogger(__name__)


def create_path_manager() -> PathManager:
    # TODO: move this inline import out after AIRStore OSS public released
    from airstore.client.airstore_tabular import AIRStorePathHandler

    pathmanager = PathManager()
    pathmanager.register_handler(AIRStorePathHandler())
    pathmanager.set_strict_kwargs_checking(False)
    return pathmanager


class AirstoreDataset(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__("airstore", config, dataset_type)

        self.pathmanager = create_path_manager()
        self.config = config
        self.batch_size = get_batch_size()
        self.airstore_uri = config.annotations.get(dataset_type)[imdb_file_index]
        self.split = dataset_type
        self.epoch = 0
        self.start_iter = 0
        self.global_rank = torch.distributed.get_rank()
        self.global_world_size = torch.distributed.get_world_size()
        self._iterator = None

    def set_epoch(self, epoch: int):
        # TODO : Currently sets the same seed every epoch, set this from MultiDataLoader
        logger.info(f"set epoch to {epoch} in airstore dataset")
        self.epoch = epoch

    def _open_iterator(self) -> Iterable[Any]:
        # iterator from airstore for current data split. data are sharded by global
        # total number of workers after shuffling

        # extract numbers of dataloading workers and current worker id (range from 0 to
        # num_workers-1) from torch.utils. If we can't get worker_info we assume the
        # current process is the only dataloading worker.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # split the dataset for each worker
        airstore_world_size = self.global_world_size * num_workers
        # each worker take it's split by it's parent process rank and worker id
        airstore_rank = self.global_rank * num_workers + worker_id
        shuffle = self.split == "train" and self.config.get("enable_shuffle", True)

        return self.pathmanager.opent(
            self.airstore_uri,
            "r",
            enable_shuffle=shuffle,
            shuffle_window=self.config.get("shuffle_window", 128),
            seed=self.epoch,
            world_size=airstore_world_size,
            rank=airstore_rank,
            limit=self.config.get("data_limit", -1),
            offset=self.config.get("data_offset", 0),
            num_of_threads=self.config.get("num_of_threads", 2),
            prefetch=self.config.get("prefetch", 1),
            max_holding_bundles=self.config.get("max_holding_bundles", 5),
            bundle_download_timeout_ms=self.config.get(
                "bundle_download_timeout_ms", 30000
            ),
            max_retries=self.config.get("max_retries", 5),
            env=self.config.get(
                "env", "OSS"
            ),  # Set to "FB" if run in FB, "RSC" for RSC, otherwise set to "OSS"
        )

    def __len__(self) -> int:
        return self._open_iterator().total_size

    def __getitem__(self, idx):
        if self._iterator is None:
            self._iterator = self._open_iterator()

        sample_info = next(self._iterator)
        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["caption"]})
        current_sample.text = processed_text["text"]
        if "input_ids" in processed_text:
            current_sample.update(processed_text)

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(BytesIO(sample_info["image"]), mode="r") as pil_img:
            image = pil_img.convert("RGB")
            current_sample.image = self.image_processor(image)

        return current_sample
