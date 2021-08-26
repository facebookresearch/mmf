# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.trainers.lightning_trainer import LightningTrainer
from tests.trainers.test_trainer_mocks import MultiDataModuleNumbersTestObject


class LightningTrainerMock(LightningTrainer):
    def __init__(self, config, num_data_size=100, **kwargs):
        super().__init__(config)

        self.config = config
        self.callbacks = []

        # settings
        trainer_config = self.config.trainer.params
        self.trainer_config = trainer_config
        self.training_config = self.config.training

        for key, value in kwargs.items():
            trainer_config[key] = value

        # data
        self.data_module = MultiDataModuleNumbersTestObject(
            config=config, num_data=num_data_size
        )

        self.run_type = self.config.get("run_type", "train")
        registry.register("config", self.config)

        self.train_loader = self.data_module.train_dataloader()
        self.val_loader = self.data_module.val_dataloader()
        self.test_loader = self.data_module.test_dataloader()
