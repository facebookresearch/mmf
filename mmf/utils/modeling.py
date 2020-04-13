from torch import nn

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
    # Pretraining has same LR for all of the parts
    if module.config.training_head_type == "pretraining":
        return get_bert_configured_parameters(module)

    # For finetuning setup, we have classifier
    lr = config.optimizer.params.lr
    model_config = getattr(config.model_config, config.model, {})
    finetune_lr_multiplier = getattr(model_config, "finetune_lr_multiplier", 1)
    # Finetune the bert pretrained part with finetune_lr_multiplier if it is set
    parameters = get_bert_configured_parameters(
        module.bert, lr * finetune_lr_multiplier
    )
    # Classifier will be trained on the normal lr
    parameters += get_bert_configured_parameters(module.classifier, lr)

    return parameters
