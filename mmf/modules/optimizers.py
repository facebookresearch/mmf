# Copyright (c) Facebook, Inc. and its affiliates.

from torch.optim import Adam
from transformers.optimization import AdamW

from mmf.common.registry import registry

registry.register_optimizer("adam")(Adam)
registry.register_optimizer("adam_w")(AdamW)
