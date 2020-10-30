---
id: training_tricks
title: Training Tricks
sidebar_label: Training Tricks
---

## FP16

MMF supports FP16 training for faster performance with negligible impact on
the results through `torch.cuda.amp` module. Append `training.fp16=True` to
the end of your command to enable fp16 training in the trainer.

## Optimizer State Sharding

MMF supports optimizer state sharding for fitting larger models in GPUs. To enable optimizer state sharding append `optimizer.enable_state_sharding=True` to the end of your command. Optimizer state sharding is achieved by using the `OSS` optimizer wrapper from [fairscale](https://github.com/facebookresearch/fairscale) library. `OSS` uses the Zero Redundancy Optimizer [(ZeRO)](https://arxiv.org/abs/1910.02054).

:::note

[fairscale](https://github.com/facebookresearch/fairscale) is not installed along with MMF due to some dependency issues. In order to use optimizer state sharding, install the library following the instructions in the repository.

:::
