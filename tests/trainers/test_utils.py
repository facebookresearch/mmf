# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.utils.build import build_optimizer
from omegaconf import OmegaConf
from tests.test_utils import SimpleLightningModel, SimpleModel
from tests.trainers.lightning.lightning_trainer_mock import LightningTrainerMock
from tests.trainers.test_trainer_mocks import TrainerTrainingLoopMock


def get_trainer_config():
    return OmegaConf.create(
        {
            "distributed": {},
            "run_type": "train",
            "training": {
                "trainer": "lightning",
                "detect_anomaly": False,
                "evaluation_interval": 4,
                "log_interval": 2,
                "update_frequency": 1,
                "fp16": False,
                "lr_scheduler": False,
                "batch_size": 1,
            },
            "evaluation": {"use_cpu": False},
            "optimizer": {"type": "adam_w", "params": {"lr": 5e-5, "eps": 1e-8}},
            "scheduler": {
                "type": "warmup_linear",
                "params": {"num_warmup_steps": 8, "num_training_steps": 8},
            },
            "trainer": {
                "type": "lightning",
                "params": {
                    "gpus": 0 if torch.cuda.is_available() else None,
                    "num_nodes": 1,
                    "precision": 32,
                    "deterministic": True,
                    "benchmark": False,
                    "gradient_clip_val": 0.0,
                    "val_check_interval": 1,
                    "log_every_n_steps": 2,
                    "checkpoint_callback": False,
                },
            },
        }
    )


def get_lightning_trainer(
    max_steps,
    max_epochs=None,
    batch_size=1,
    model_size=1,
    accumulate_grad_batches=1,
    callback=None,
    lr_scheduler=False,
    gradient_clip_val=0.0,
    precision=32,
):
    torch.random.manual_seed(2)
    trainer = LightningTrainerMock(
        config=get_trainer_config(),
        max_steps=max_steps,
        max_epochs=max_epochs,
        callback=callback,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        lr_scheduler=lr_scheduler,
        gradient_clip_val=gradient_clip_val,
        precision=precision,
    )
    trainer.model = SimpleLightningModel(
        {"in_dim": model_size}, trainer_config=trainer.config
    )
    trainer.model.build()
    trainer.model.train()
    trainer.model.is_pl_enabled = True
    prepare_lightning_trainer(trainer)
    return trainer


def get_mmf_trainer(
    model_size=1,
    num_data_size=100,
    max_updates=5,
    max_epochs=None,
    on_update_end_fn=None,
    fp16=False,
    scheduler_config=None,
    grad_clipping_config=None,
):
    torch.random.manual_seed(2)
    model = SimpleModel({"in_dim": model_size})
    model.build()
    model.train()
    trainer_config = get_trainer_config()
    optimizer = build_optimizer(model, trainer_config)
    trainer = TrainerTrainingLoopMock(
        num_data_size,
        max_updates,
        max_epochs,
        config=trainer_config,
        optimizer=optimizer,
        on_update_end_fn=on_update_end_fn,
        fp16=fp16,
        scheduler_config=scheduler_config,
        grad_clipping_config=grad_clipping_config,
    )
    trainer.load_datasets()
    model.to(trainer.device)
    trainer.model = model
    return trainer


def prepare_lightning_trainer(trainer):
    trainer.configure_device()
    trainer._calculate_max_updates()
    trainer._load_trainer()
