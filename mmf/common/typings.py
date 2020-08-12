# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import omegaconf
import torch
from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.utils.download import DownloadableFile


DownloadableFileType = Type[DownloadableFile]
DatasetType = Type[BaseDataset]
DatasetBuilderType = Type[BaseDatasetBuilder]
DictConfig = Type[omegaconf.DictConfig]
DataLoaderAndSampler = Tuple[
    Type[torch.utils.data.DataLoader], Optional[torch.utils.data.Sampler]
]
DataLoaderArgsType = Optional[Dict[str, Any]]


@dataclass
class PerSetAttributeType:
    train: List[str]
    val: List[str]
    test: List[str]


@dataclass
class ProcessorConfigType:
    type: str
    params: Dict[str, Any]


@dataclass
class MMFDatasetConfigType:
    data_dir: str
    use_images: bool
    use_features: bool
    zoo_requirements: List[str]
    images: PerSetAttributeType
    features: PerSetAttributeType
    annotations: PerSetAttributeType
    processors: Dict[str, ProcessorConfigType]
