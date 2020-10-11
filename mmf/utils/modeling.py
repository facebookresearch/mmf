# Copyright (c) Facebook, Inc. and its affiliates.

import logging

from torch import nn


logger = logging.getLogger(__name__)

ACT2FN = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def get_bert_configured_parameters(module, lr=None):
    param_optimizer = list(module.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if lr is not None:
        for p in optimizer_grouped_parameters:
            p["lr"] = lr

    return optimizer_grouped_parameters


def get_optimizer_parameters_for_bert(module, config):
    lr = config.optimizer.params.lr
    model_config = getattr(config.model_config, config.model, {})
    finetune_lr_multiplier = getattr(model_config, "finetune_lr_multiplier", 1)

    # For pretraining or when finetune_lr_multiplier == 1, all modules will be trained
    # with default lr.
    if module.config.training_head_type == "pretraining" or finetune_lr_multiplier == 1:
        return get_bert_configured_parameters(module)

    # For non pretraining heads, where finetune_lr_multiplier != 1, all modules other
    # than classifier will be trained with (lr * finetune_lr_multiplier).
    parameters = []
    for name, submodule in module.named_children():
        if name == "classifier":
            continue
        parameters += get_bert_configured_parameters(
            submodule, lr * finetune_lr_multiplier
        )
        logger.info(f"Overriding {name} module's LR to {lr * finetune_lr_multiplier}")
    # Classifier will be trained with default lr.
    parameters += get_bert_configured_parameters(module.classifier)

    return parameters
