# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Type

from mmf.common import typings as mmf_typings
from mmf.trainers.base_trainer import BaseTrainer


class Callback:
    """
    Base class for callbacks that can be registered with type :class:`BaseTrainer`

    Attr:
        config(mmf_typings.DictConfig): Config for the callback
        trainer(Type[BaseTrainer]): Trainer object
    """

    def __init__(
        self, config: mmf_typings.DictConfig, trainer: Type[BaseTrainer]
    ) -> None:
        self.config = config
        self.trainer = trainer
        self.training_config = self.config.training

    def on_init_start(self, **kwargs) -> None:
        """
        Called when the trainer initialization begins.
        """
        pass

    def on_init_end(self, **kwargs) -> None:
        """
        Called when the trainer initialization ends.
        """
        pass

    def on_train_start(self, **kwargs) -> None:
        """
        Called before training starts.
        """
        pass

    def on_train_end(self, **kwargs) -> None:
        """
        Called after training ends.
        """
        pass

    def on_batch_start(self, **kwargs) -> None:
        """
        Called before each train iteration.
        """
        pass

    def on_batch_end(self, **kwargs) -> None:
        """
        Called after each train iteration.
        """
        pass

    def on_validation_start(self, **kwargs) -> None:
        """
        Called before validation starts.
        """
        pass

    def on_validation_end(self, **kwargs) -> None:
        """
        Called after validation ends.
        """
        pass

    def on_validation_batch_start(self, **kwargs) -> None:
        """
        Called before each validation iteration.
        """
        pass

    def on_validation_batch_end(self, **kwargs) -> None:
        """
        Called after each validation iteration.
        """
        pass

    def on_test_start(self, **kwargs) -> None:
        """
        Called before test starts.
        """
        pass

    def on_test_end(self, **kwargs) -> None:
        """
        Called after test ends.
        """
        pass

    def on_test_batch_start(self, **kwargs) -> None:
        """
        Called before each test iteration.
        """
        pass

    def on_test_batch_end(self, **kwargs) -> None:
        """
        Called after each test iteration.
        """
        pass
