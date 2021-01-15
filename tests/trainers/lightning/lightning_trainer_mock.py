# Copyright (c) Facebook, Inc. and its affiliates.


from unittest.mock import MagicMock

import torch
from mmf.trainers.lightning_trainer import LightningTrainer
from tests.test_utils import NumbersDataset


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
        **kwargs,
    ):
        self.config = config
        self._callbacks = None
        if callback:
            self._callbacks = [callback]

        # data
        self.data_module = MagicMock()
        train_dataset = NumbersDataset(num_data_size)
        val_dataset = NumbersDataset(num_data_size)

        self.data_module.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )
        self.data_module.train_loader.change_dataloader = MagicMock(return_value=None)
        self.data_module.val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )
        self.data_module.val_loader.change_dataloader = MagicMock(return_value=None)
        self.data_module.train_loader.current_dataset = train_dataset
        self.data_module.val_loader.current_dataset = val_dataset

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
        self.run_type = self.config.run_type
        self.config.training.lr_scheduler = lr_scheduler
