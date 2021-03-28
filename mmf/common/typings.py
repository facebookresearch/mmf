# Copyright (c) Facebook, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Any, Dict, List


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
