# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC
from typing import List

from mmf.trainers.callbacks.base import Callback


class TrainerCallbackHookMixin(ABC):
    callbacks: List[Callback] = []

    def on_init_start(self, **kwargs) -> None:
        """Called when the trainer initialization begins."""
        for callback in self.callbacks:
            callback.on_init_start(**kwargs)

    def on_init_end(self, **kwargs) -> None:
        """Called when the trainer initialization ends."""
        for callback in self.callbacks:
            callback.on_init_end(**kwargs)

    def on_train_start(self, **kwargs) -> None:
        """Called when training begins."""
        for callback in self.callbacks:
            callback.on_train_start(**kwargs)

    def on_train_end(self, **kwargs) -> None:
        """Called when training ends."""
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_batch_start(self, **kwargs) -> None:
        """Called when a forward pass begins."""
        for callback in self.callbacks:
            callback.on_batch_start(**kwargs)

    def on_batch_end(self, **kwargs) -> None:
        """Called when a forward pass ends."""
        for callback in self.callbacks:
            callback.on_batch_end(**kwargs)

    def on_update_start(self, **kwargs) -> None:
        """Called when the training update begins."""
        for callback in self.callbacks:
            callback.on_update_start(**kwargs)

    def on_update_end(self, **kwargs) -> None:
        """Called when the training update ends."""
        for callback in self.callbacks:
            callback.on_update_end(**kwargs)

    def on_validation_start(self, **kwargs) -> None:
        """Called when the validation loop begins."""
        for callback in self.callbacks:
            callback.on_validation_start(**kwargs)

    def on_validation_end(self, **kwargs) -> None:
        """Called when the validation loop ends."""
        for callback in self.callbacks:
            callback.on_validation_end(**kwargs)

    def on_validation_batch_start(self, **kwargs) -> None:
        """Called when the validation batch begins."""
        for callback in self.callbacks:
            callback.on_validation_batch_start(**kwargs)

    def on_validation_batch_end(self, **kwargs) -> None:
        """Called when the validation batch ends."""
        for callback in self.callbacks:
            callback.on_validation_batch_end(**kwargs)

    def on_test_start(self, **kwargs) -> None:
        """Called when the test begins."""
        for callback in self.callbacks:
            callback.on_test_start(**kwargs)

    def on_test_end(self, **kwargs) -> None:
        """Called when the test ends."""
        for callback in self.callbacks:
            callback.on_test_end(**kwargs)

    def on_test_batch_start(self, **kwargs) -> None:
        """Called when the test batch begins."""
        for callback in self.callbacks:
            callback.on_test_batch_start(**kwargs)

    def on_test_batch_end(self, **kwargs) -> None:
        """Called when the test batch ends."""
        for callback in self.callbacks:
            callback.on_test_batch_end(**kwargs)

    def on_prediction_start(self, **kwargs) -> None:
        """Called when the prediction begins."""
        for callback in self.callbacks:
            callback.on_prediction_start(**kwargs)

    def on_prediction_end(self, **kwargs) -> None:
        """Called when the prediction ends."""
        for callback in self.callbacks:
            callback.on_prediction_end(**kwargs)
