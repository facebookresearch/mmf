# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.utils.build import build_optimizer
from omegaconf import OmegaConf
from tests.test_utils import SimpleLightningModel, SimpleModel
from tests.trainers.lightning.lightning_trainer_mock import LightningTrainerMock
from tests.trainers.test_training_loop import TrainerTrainingLoopMock


def get_trainer_config():
    return OmegaConf.create(
        {
            "distributed": {},
            "run_type": "train_inference",
            "training": {
                "trainer": "lightning",
                "detect_anomaly": False,
                "evaluation_interval": 4,
                "log_interval": 100,
                "update_frequency": 1,
                "fp16": False,
                "batch_size": 1,
                "lr_scheduler": False,
                "tensorboard": False,
            },
            "optimizer": {"type": "adam_w", "params": {"lr": 5e-5, "eps": 1e-8}},
            "scheduler": {
                "type": "warmup_linear",
                "params": {"num_warmup_steps": 8, "num_training_steps": 8},
            },
            "evaluation": {"metrics": []},
            "trainer": {
                "type": "lightning",
                "params": {
                    "gpus": [0] if torch.cuda.is_available() else None,
                    "num_nodes": 1,
                    "gradient_clip_val": 0.0,
                    "checkpoint_callback": False,
                    "deterministic": True,
                    "benchmark": False,
                    "accumulate_grad_batches": 1,
                    "log_every_n_steps": 100,
                    "limit_val_batches": 1.0,
                    "progress_bar_refresh_rate": 0,
                    "precision": 32,
                    "val_check_interval": 4,
                    "num_sanity_val_steps": 0,
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
    prepare_trainer=True,
    tensorboard=False,
    **kwargs,
):
    torch.random.manual_seed(2)
    trainer_config = get_trainer_config()
    trainer_config.training.tensorboard = tensorboard
    trainer = LightningTrainerMock(
        config=trainer_config,
        max_steps=max_steps,
        max_epochs=max_epochs,
        callback=callback,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        lr_scheduler=lr_scheduler,
        gradient_clip_val=gradient_clip_val,
        precision=precision,
        **kwargs,
    )
    trainer.model = SimpleLightningModel(model_size, config=trainer.config)
    trainer.model.train()
    if prepare_trainer:
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
    evaluation_interval=4,
    batch_size=1,
    mock_functions=True,
    tensorboard=False,
):
    torch.random.manual_seed(2)
    model = SimpleModel(model_size)
    model.train()
    trainer_config = get_trainer_config()
    trainer_config.training.evaluation_interval = evaluation_interval
    optimizer = build_optimizer(model, trainer_config)
    trainer = TrainerTrainingLoopMock(
        num_data_size,
        max_updates,
        max_epochs,
        config=trainer_config,
        optimizer=optimizer,
        fp16=fp16,
        scheduler_config=scheduler_config,
        grad_clipping_config=grad_clipping_config,
        batch_size=batch_size,
        mock_functions=mock_functions,
        tensorboard=tensorboard,
    )
    if on_update_end_fn:
        trainer.on_update_end = on_update_end_fn
    model.to(trainer.device)
    trainer.model = model
    return trainer


def prepare_lightning_trainer(trainer):
    trainer.configure_device()
    trainer._calculate_max_updates()
    trainer.load_metrics()
    trainer._load_loggers()
    trainer._load_trainer()


def run_lightning_trainer_with_callback(trainer, callback, on_fit_start_callback=None):
    callback.lightning_trainer = trainer
    callback.trainer_config = trainer.trainer_config
    callback.training_config = trainer.training_config
    trainer._callbacks = [callback]

    prepare_lightning_trainer(trainer)
    if on_fit_start_callback:
        on_fit_start_callback()
    trainer.trainer.fit(
        trainer.model,
        train_dataloader=trainer.data_module.train_loader,
        val_dataloaders=trainer.data_module.val_loader,
    )
