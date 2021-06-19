# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import torch
from tests.test_utils import SimpleModel, skip_if_no_cuda
from tests.trainers.test_training_loop import TrainerTrainingLoopMock
from tests.trainers.test_utils import get_config_with_defaults


class SimpleModelWithFp16Assert(SimpleModel):
    def forward(self, sample_list):
        batch_tensor = sample_list[list(sample_list.keys())[0]]
        # Start should be fp32
        assert batch_tensor.dtype == torch.float32
        batch_tensor = self.classifier(batch_tensor)

        # In between operation should be fp16
        assert batch_tensor.dtype == torch.float16
        loss = torch.sum(batch_tensor)

        # Sum should convert it back to fp32
        assert loss.dtype == torch.float32

        model_output = {"losses": {"loss": loss}}
        return model_output


class MMFTrainerMock(TrainerTrainingLoopMock):
    def __init__(
        self, num_train_data, max_updates, max_epochs, device="cuda", fp16_model=False
    ):
        config = get_config_with_defaults(
            {
                "training": {
                    "max_updates": max_updates,
                    "max_epochs": max_epochs,
                    "evaluation_interval": 10000,
                    "fp16": True,
                },
                "run_type": "train",
            }
        )
        super().__init__(num_train_data, config=config)
        if fp16_model:
            assert (
                torch.cuda.is_available()
            ), "MMFTrainerMock fp16 requires cuda enabled"
            model = SimpleModelWithFp16Assert({"in_dim": 1})
            model.build()
            model = model.cuda()
        else:
            model = SimpleModel({"in_dim": 1})
            model.build()
            model.train()
            model.to(self.device)

        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)


class TestFp16(unittest.TestCase):
    @skip_if_no_cuda
    def test_fp16_works(self):
        trainer = MMFTrainerMock(100, 2, 0.04)
        trainer.load_datasets()
        trainer.load_fp16_scaler()
        self.assertTrue(isinstance(trainer.scaler, torch.cuda.amp.GradScaler))
        self.assertEqual(trainer.current_iteration, 0)
        trainer.training_loop()
        self.assertEqual(trainer.current_iteration, 4)

    @skip_if_no_cuda
    def test_fp16_values(self):
        trainer = MMFTrainerMock(100, 2, 0.04, fp16_model=True)
        trainer.load_datasets()
        trainer.load_fp16_scaler()
        trainer.training_loop()
