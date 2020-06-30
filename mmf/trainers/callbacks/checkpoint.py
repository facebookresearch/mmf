# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.trainers.callbacks.base import Callback
from mmf.utils.checkpoint import Checkpoint


class CheckpointCallback(Callback):
    """Callback for executing different checkpoint requirements.
    """

    def __init__(self, config, trainer):
        """
        Attr:
            config(mmf_typings.DictConfig): Config for the callback
            trainer(Type[BaseTrainer]): Trainer object
        """
        super().__init__(config, trainer)

        self._checkpoint = Checkpoint(trainer)
        self.checkpoint_interval = self.config.training.checkpoint_interval
        self.writer = registry.get("writer")

    @property
    def checkpoint(self):
        return self._checkpoint

    def on_init_end(self, **kwargs):
        self._checkpoint.load_state_dict()

    def on_batch_end(self, **kwargs):
        if self.trainer.num_updates % self.checkpoint_interval == 0:
            self.writer.write("Checkpoint time. Saving a checkpoint.")
            self._checkpoint.save(
                self.trainer.num_updates,
                self.trainer.current_iteration,
                update_best=False,
            )

    def on_train_end(self, **kwargs):
        self._checkpoint.restore()
        self._checkpoint.finalize()
