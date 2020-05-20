# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any, Dict, Optional, Tuple, Type

import omegaconf
import torch

from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder

DatasetType = Type[BaseDataset]
DatasetBuilderType = Type[BaseDatasetBuilder]
DictConfig = Type[omegaconf.DictConfig]
DataLoaderAndSampler = Tuple[
    Type[torch.utils.data.DataLoader], Optional[torch.utils.data.Sampler]
]
DataLoaderArgsType = Optional[Dict[str, Any]]
