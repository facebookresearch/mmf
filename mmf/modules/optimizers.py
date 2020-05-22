# Copyright (c) Facebook, Inc. and its affiliates.

from transformers.optimization import AdamW

from mmf.common.registry import registry

registry.register_optimizer("adam_w")(AdamW)
