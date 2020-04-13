# Copyright (c) Facebook, Inc. and its affiliates.
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from mmf.common.registry import registry


@registry.register_scheduler("pythia")
class PythiaScheduler(LambdaLR):
    def __init__(self, optimizer, *args, **kwargs):
        from mmf.utils.general import lr_lambda_update

        self._lambda_func = lr_lambda_update
        self._global_config = registry.get("config")

        super().__init__(optimizer, self.lr_lambda, *args, **kwargs)

    def lr_lambda(self, step):
        return self._lambda_func(step, self._global_config)


@registry.register_scheduler("warmup_linear")
class WarmupLinearScheduler(LambdaLR):
    def __new__(cls, optimizer, *args, **kwargs):
        return get_linear_schedule_with_warmup(optimizer, *args, **kwargs)


@registry.register_scheduler("warmup_cosine")
class WarmupCosineScheduler(LambdaLR):
    def __new__(cls, optimizer, *args, **kwargs):
        return get_cosine_schedule_with_warmup(optimizer, *args, **kwargs)
