# Copyright (c) Facebook, Inc. and its affiliates.
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from pytorch_transformers.optimization import WarmupLinearSchedule

from pythia.common.registry import registry


@registry.register_scheduler("pythia")
class PythiaScheduler(LambdaLR):
    def __init__(self, optimizer, *args, **kwargs):
        from pythia.utils.general import lr_lambda_update
        self._lambda_func = lr_lambda_update
        self._global_config = registry.get("config")

        super().__init__(optimizer, self.lr_lambda, *args, **kwargs)

    def lr_lambda(self, step):
        return self._lambda_func(step, self._global_config)


registry.register_scheduler("warmup_linear")(WarmupLinearSchedule)
