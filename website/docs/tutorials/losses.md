---
id: losses
title: Adding a custom loss
sidebar_label: Adding a custom loss
---

# Custom Losses

This is a tutorial on how to add a new loss function to MMF.

MMF is agnostic to the kind of losses that can be added to it.
Adding a loss requires adding a loss class and adding your new loss to your config yaml.
For example, the [ConcatBERT](https://github.com/facebookresearch/mmf/blob/main/website/docs/tutorials/concat_bert_tutorial.md) model uses the `cross_entropy` loss when training on the hateful memes dataset.
The loss class is `CrossEntropyLoss` defined in [mmf/modules/losses.py](https://github.com/facebookresearch/mmf/blob/main/mmf/modules/losses.py)
The loss key `cross_entropy` is added to the list of losses in the config yaml at [mmf/projects/hateful_memes/configs/concat_bert/defaults.yaml](https://github.com/facebookresearch/mmf/blob/15fa63071bfaed56db43deba871cfec76439c66f/projects/others/concat_bert/hateful_memes/defaults.yaml#L11).


# Loss Class

Add your loss class to losses.py. It should be a subclass of `nn.Module`.
Losses should implement a function forward with signature `forward(self, sample_list, model_output)`,
where sample_list (`SampleList`) is the current batch and model_output is a dict return by your model for current sample_list.

```python
@registry.register_loss("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**params)

    def forward(self, sample_list, model_output):
        return self.loss_fn(model_output["scores"], sample_list["targets"])
```

# Losses Config

Add the name of your new loss class to your model config.
Multiple losses can be specified with a yaml array.

```yaml
model_config:
  visual_bert:
    training_head_type: classification
    num_labels: 2
    losses:
    - cross_entropy
    - soft_label_cross_entropy
    ...
```

For losses with params you can do,

```yaml
    losses:
    - type: in_batch_hinge
      params:
        margin: 0.2
        hard: true
```

# Multi-Loss Classes

If a loss class is responsible for calculating multiple losses, for example, maybe due to shared calculations you can return a dictionary of tensors.
The resulting loss that is optimized is the sum of all losses configured for the model.
For an example, take a look at the `BCEAndKLLoss` class in [mmf/modules/losses.py](https://github.com/facebookresearch/mmf/blob/main/mmf/modules/losses.py)

```python
@registry.register_loss("bce_kl")
class BCEAndKLLoss(nn.Module):
    """binary_cross_entropy_with_logits and kl divergence loss.
    Calculates both losses and returns a dict with string keys.
    Similar to bce_kl_combined, but returns both losses.
    """

    def __init__(self, weight_softmax):
        super().__init__()
        self.weight_softmax = weight_softmax

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss1 = kl_div(res, tar)
        loss1 = torch.sum(loss1) / loss1.size(0)

        loss2 = F.binary_cross_entropy_with_logits(
            pred_score, target_score, reduction="mean"
        )
        loss2 *= target_score.size(1)

        loss = {"kl": self.weight_softmax * loss1, "bce": loss2}

        return loss
```
