# Copyright (c) Facebook, Inc. and its affiliates.

import os

import torch
from mmf.utils.build import build_optimizer
from mmf.utils.configuration import load_yaml
from omegaconf import OmegaConf
from tests.test_utils import SimpleLightningModel, SimpleModel
from tests.trainers.lightning.lightning_trainer_mock import LightningTrainerMock
from tests.trainers.test_trainer_mocks import TrainerTrainingLoopMock


def get_trainer_config():
    config = load_yaml(os.path.join("configs", "defaults.yaml"))
    return OmegaConf.merge(
        config,
        {
            "distributed": {},
            "run_type": "train_val",
            "training": {
                "trainer": "lightning",
                "detect_anomaly": False,
                "evaluation_interval": 4,
                "log_interval": 2,
                "update_frequency": 1,
                "fp16": False,
                "batch_size": 1,
                "batch_size_per_device": None,
                "lr_scheduler": False,
                "tensorboard": False,
                "num_workers": 0,
                "max_grad_l2_norm": 1,
                "exit_on_nan_losses": True,
            },
            "optimizer": {"type": "adam_w", "params": {"lr": 5e-5, "eps": 1e-8}},
            "scheduler": {
                "type": "warmup_linear",
                "params": {"num_warmup_steps": 8, "num_training_steps": 8},
            },
            "trainer": {
                "type": "lightning",
                "params": {
                    "gpus": 1 if torch.cuda.is_available() else 0,
                    "num_nodes": 1,
                    "enable_checkpointing": False,
                    "deterministic": True,
                    "benchmark": False,
                    "gradient_clip_val": 0.0,
                    "val_check_interval": 4,
                    "log_every_n_steps": 2,
                    "progress_bar_refresh_rate": 0,
                    "accumulate_grad_batches": 1,
                    "precision": 32,
                    "num_sanity_val_steps": 0,
                    "limit_val_batches": 1.0,
                    "logger": False,
                },
            },
        },
    )


def get_config_with_defaults(new_config):
    config = get_trainer_config()
    config = OmegaConf.merge(config, OmegaConf.create(new_config))
    return config


def add_model(trainer, model):
    model.build()
    model.train()
    model.to(trainer.device)
    trainer.model = model


def add_optimizer(trainer, config):
    optimizer = build_optimizer(trainer.model, config)
    trainer.optimizer = optimizer


def get_mmf_trainer(
    config=None, model_size=1, num_data_size=100, load_model_from_config=False, seed=2
):
    torch.random.manual_seed(seed)
    trainer = TrainerTrainingLoopMock(num_data_size, config=config)

    if not load_model_from_config:
        add_model(trainer, SimpleModel({"in_dim": model_size}))
    else:
        trainer.load_model()

    add_optimizer(trainer, config)

    trainer.load_datasets()
    return trainer


def get_lightning_trainer(
    config=None,
    model_size=1,
    prepare_trainer=True,
    load_model_from_config=False,
    seed=2,
    **kwargs,
):
    torch.random.manual_seed(seed)
    trainer = LightningTrainerMock(config=config, **kwargs)

    if not load_model_from_config:
        trainer.model = SimpleLightningModel({"in_dim": model_size})
        trainer.model.build()
        trainer.model.train()

        trainer.model.build_meters(trainer.run_type)
        trainer.model.is_pl_enabled = True
    else:
        trainer.load_model()

    if prepare_trainer:
        prepare_lightning_trainer(trainer)

    return trainer


def prepare_lightning_trainer(trainer):
    trainer.configure_device()
    trainer._calculate_max_updates()
    trainer.load_metrics()
    trainer._load_loggers()
    trainer._load_trainer()


def run_lightning_trainer(trainer, on_fit_start_callback=None):
    prepare_lightning_trainer(trainer)
    if on_fit_start_callback:
        on_fit_start_callback()

    trainer.trainer.fit(
        trainer.model,
        train_dataloaders=trainer.train_loader,
        val_dataloaders=trainer.val_loader,
    )
    trainer.run_last_validation_after_train()
