---
id: checkpointing
title: 'Tutorial: Understanding Checkpointing for Pretraining and Finetuning'
sidebar_label: Checkpointing
---

In this tutorial, we will learn about the different details around finetuning from pretrained models like loading from checkpoints, loading a model from the model zoo and doing validation/inference using finetuned models. We will walk-through the tutorial by training a Visual BERT model and train/validate on Hateful Memes dataset.

## Pre-requisites and installation

Follow the prerequisites for installation and dataset [here](https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes#prerequisites).

## Finetuning from a pretrained model

VisualBERT model is pretrained on V+L multimodal data. We will use a pretrained model on COCO Captions. To begin finetuning our VisualBERT model we will load a model pretrained on COCO Captions and finetune that on Hateful Memes.

```bash
mmf_run config=projects/visual_bert/configs/hateful_memes/from_coco.yaml \
    model=visual_bert \
    dataset=hateful_memes \
    run_type=train_val
```

The config file contains two important changes :

```yaml
checkpoint:
  resume_pretrained: true
  resume_zoo: visual_bert.pretrained.coco
```

Here `checkpoint.resume_pretrained` specifies if we want to resume from a pretrained model using the pretrained state dict mappings defined in `checkpoint.pretrained_state_mapping`. `checkpoint.resume_zoo` specifies which pretrained model from our model zoo we want to use for this. In this case, we will use `visual_bert.pretrained.coco`.

`checkpoint.pretrained_state_mapping` specifies how a pretrained model will be loaded and mapped to which keys of the target model. We use it since we only want to load specific layers from the pretrained model. In the case of VisualBERT model, we want to load the pretrained `bert` layers. This is specified in our defaults.yaml:

```yaml
checkpoint:
  pretrained_state_mapping:
    model.bert: model.bert
```

This will ensure only the `model.bert` layers of the COCO pretrained model gets loaded.

We can also use the default config for VisualBERT on hateful memes directly and override the pretrained options through [command line args](https://mmf.sh/docs/notes/configuration#command-line-dot-list-override):

```
mmf_run config=projects/visual_bert/configs/hateful_memes/defaults.yaml model=visual_bert dataset=hateful_memes \
run_type=train_val checkpoint.resume_pretrained=True checkpoint.resume_zoo=visual_bert.pretrained.coco
```

After running the training our model will be saved in `./save/<experiment_name>/visual_bert_final.pth`. Replace `./save` with `env.save_dir` if overriden. This will be the directory structure:

```bash
├── best.ckpt
├── config.yaml
├── current.ckpt
├── logs
├── models
├── train.log
├── visual_bert_final.pth
```

## Finetuning from a pretrained checkpoint

Instead of loading from the model zoo we can also load from a file:

```bash
mmf_run config=projects/visual_bert/configs/hateful_memes/defaults.yaml \
    model=visual_bert \
    dataset=hateful_memes \
    run_type=train_val \
    checkpoint.resume_pretrained=True \
    checkpoint.resume_file=<path_to_your_pretrained_model>
```

`checkpoint.resume_file` can also be used when loading a model file for evaluation or generating predictions. We will see more example usage of this later.

## Resuming training

To resume the training in case it gets intterupted, run:

```bash
mmf_run config=projects/visual_bert/configs/hateful_memes/defaults.yaml \
    model=visual_bert \
    dataset=hateful_memes \
    run_type=train_val \
    checkpoint.resume=True
```

When `checkpoint.resume=True`, MMF will try to load automatically the last saved checkpoint in the `env.save_dir` experiment folder `current.ckpt`.

Instead of the last saved checkpoint, we can also resume from the "best" checkpoint based on `training.early_stop.criteria` if enabled in the config. This can be achieved using `checkpoint.resume_best=True`:

```bash
mmf_run config=projects/visual_bert/configs/hateful_memes/defaults.yaml \
    model=visual_bert \
    dataset=hateful_memes \
    run_type=train_val \
    checkpoint.resume=True \
    checkpoint.resume_best=True
```

In the config early stopping parameters are as follows:

```yaml
training:
  early_stop:
    # Criteria(loss or metric) to be monitored for early stopping
    criteria: hateful_memes/roc_auc
    # Whether the monitored criteria should be minimized
    minimize: false
```

## Limiting the Number of Saved Checkpoints
Optionally, you can limit the maximum number of checkpoint files that are saved.
In the config, this is managed with the parameter `max_to_keep` under `checkpoint`.
```yaml
checkpoint:
  max_to_keep: -1
```
When the parameter is set to -1, every eligible checkpoint is saved; otherwise, only the last `max_to_keep` checkpoints are kept at any given time.

## Running validation using the trained model

After we finish the training we will load the trained model for validation:

```bash
mmf_run config=projects/visual_bert/configs/hateful_memes/defaults.yaml \
    model=visual_bert \
    dataset=hateful_memes \
    run_type=val \
    checkpoint.resume_file=<path_to_finetuned_model>
```

Note that here we specify `run_type=val` so that we are running only validation. We also use `checkpoint.resume_file` to load the trained model.

## Generating predictions using the trained model

We will next load the trained model to generate prediction results:

```bash
mmf_predict config=projects/visual_bert/configs/hateful_memes/defaults.yaml \
    model=visual_bert \
    dataset=hateful_memes \
    run_type=test \
    checkpoint.resume_file=<path_to_finetuned_model>
```

This will generate a submission file in csv format that can be used for submission to the Hateful Memes challenge.
