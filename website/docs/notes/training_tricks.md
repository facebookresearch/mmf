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

## Splitting Dataset

MMF supports spliting dataset dynamically. For example, users of MMF might want to split some percentage of the train dataset to be used for validation or test. [#470](https://github.com/facebookresearch/mmf/pull/470) introduced such a feature. You can specifiy how much of the train dataset will be used for eval or test with the following config (textvqa dataset as an example):

```yaml
 textvqa:
    zoo_requirements:
    - textvqa.defaults
    - textvqa.ocr_en
    split_train:
      val: 0.99 # 0.99 of the train dataset will be used for validation
      test: 0.001 # 0.001 of the train dataset will be used for test
      seed: 123456 # this is the default seed used for the random split. This line is optional.
    features:
      train:
      - textvqa/defaults/features/open_images/detectron.lmdb,textvqa/ocr_en/features/ocr_en_frcn_features.lmdb
      test:
      - test_path
   processors:
      ...
```

:::note

val and test percentages must be less than 100%. There should to be some percentages left for the training dataset.

:::

