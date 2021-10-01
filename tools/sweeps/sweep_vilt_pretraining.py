#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

import lib as sweep
from lib import hyperparam


def get_grid(args):

    return [
        hyperparam("run_type", "train_val"),
        hyperparam("config", "projects/vilt/configs/cmd/rand_init_training.yaml"),
        # hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("dataset", "cmd"),
        hyperparam("model", "vilt", save_dir_key=lambda val: val),
        # For nlvr2, we are able to fit batch of size 16 on single GPU with 16GB
        # memory. Same number is 32 for VQA2, so scale accordingly
        hyperparam("training.batch_size", 2048, save_dir_key=lambda val: f"bs{val}"),
        hyperparam("training.seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("training.num_workers", 1),
        # hyperparam("scheduler.type", ["warmup_cosine"]),
        # hyperparam("scheduler.params.num_warmup_steps", 2000),
        # hyperparam("scheduler.params.num_training_steps", max_update),
        # hyperparam("optimizer.type", "adam_w", save_dir_key=lambda val: val),
        # hyperparam(
        #     "optimizer.params.lr", [5e-5, 1e-5], save_dir_key=lambda val: f"lr{val}"
        # ),
        # hyperparam("optimizer.params.eps", 1e-8),
        hyperparam("training.max_updates", 100000, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("training.log_format", "json"),
        hyperparam("training.pin_memory", True),
        hyperparam("training.log_interval", 10000),
        # hyperparam("training.checkpoint_interval", 1000),
        # hyperparam("training.evaluation_interval", 4000),
        hyperparam("training.find_unused_parameters", True),
        hyperparam("env.user_dir", "/private/home/ryanjiang/megavlt"),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
