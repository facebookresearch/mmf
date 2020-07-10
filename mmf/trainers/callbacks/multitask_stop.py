# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.trainers.callbacks.base import Callback
from mmf.utils.distributed import broadcast_scalar

class MultitaskStopCallback(Callback):
    """Callback for Multitask stop plateu mechanism and checks if it training
    should continue or stop.
    """

    def __init__(self, config, trainer):
        """
        Attr:
            config(mmf_typings.DictConfig): Config for the callback
            trainer(Type[BaseTrainer]): Trainer object
        """
        super().__init__(config, trainer)






