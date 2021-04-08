# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.trainers.lightning_trainer import LightningTrainer
from tests.trainers.test_trainer_mocks import MultiDataModuleNumbersTestObject


class LightningTrainerMock(LightningTrainer):
    def __init__(
        self,
        config,
        max_steps,
        max_epochs=None,
        callback=None,
        num_data_size=100,
        batch_size=1,
        accumulate_grad_batches=1,
        lr_scheduler=False,
        gradient_clip_val=0.0,
        precision=32,
        **kwargs
    ):
        self.config = config
        self._callbacks = None
        if callback:
            self._callbacks = [callback]

        # data
        self.data_module = MultiDataModuleNumbersTestObject(
            num_data=num_data_size, batch_size=batch_size
        )

        # settings
        trainer_config = self.config.trainer.params
        trainer_config.accumulate_grad_batches = accumulate_grad_batches
        trainer_config.precision = precision
        trainer_config.max_steps = max_steps
        trainer_config.max_epochs = max_epochs
        trainer_config.gradient_clip_val = gradient_clip_val
        trainer_config.precision = precision

        for key, value in kwargs.items():
            trainer_config[key] = value

        self.trainer_config = trainer_config
        self.training_config = self.config.training
        self.training_config.batch_size = batch_size
        self.run_type = self.config.get("run_type", "train")
        self.config.training.lr_scheduler = lr_scheduler
        registry.register("config", self.config)

        self.data_module = MultiDataModuleNumbersTestObject(
            num_data=num_data_size, batch_size=batch_size
        )
        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        self.test_loader = self.data_module.test_dataloader()
