# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, Optional

from torch import Tensor


def getattr_torchscriptable(
    dictionary: Dict[str, Tensor], key: str, default: Optional[Tensor] = None
) -> Optional[Tensor]:
    if key in dictionary:
        return dictionary[key]
    else:
        return default
