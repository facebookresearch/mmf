---
id: hateful_memes_challenge
title: Hateful Memes Challenge
sidebar_label: Hateful Memes Challenge
---

The Hateful Memes challenge is available at [this link](https://www.drivendata.org/competitions/70/hateful-memes-phase-2/data/).

In MMF, we provide the starter code and baseline pretrained models for this challenge and the configurations used for training the reported baselines. For more details check [this link](https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes).

In this tutorial, we provide steps for running training and evaluation with MMBT model on hateful memes dataset and generating submission file for the challenge. The same steps can be used for your own models.

## Installation and Preparing the dataset

Follow the prerequisites for installation and dataset [here](https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes#prerequisites).

## Training and Evaluation

### Training

For running training on train set, run the following command:

```bash
mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml \
    model=mmbt \
    dataset=hateful_memes \
    run_type=train_val
```

This will train the `mmbt` model on the dataset and generate the checkpoints and best trained model (`mmbt_final.pth`) will be stored in the `./save` directory by default.

### Evaluation

Next run evaluation on the validation set:

```bash
mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml \
    model=mmbt \
    dataset=hateful_memes \
    run_type=val \
    checkpoint.resume_file=./save/mmbt_final.pth \
    checkpoint.resume_pretrained=False
```

This will give you the performance of your model on the validation set. The metrics are AUROC, ACC, Binary F1 etc.

## Predictions for Challenge

After we trained the model and evaluated on the validation set, we will generate the predictions on the test set. The prediction file should contain the following three columns:

- Meme identification number, `id`
- Probability that the meme is hateful, `proba`
- Binary label that the meme is hateful (1) or non-hateful (0), `label`

With MMF you can directly generate the predictions in the required submission format with the following command:

```bash
mmf_predict config=projects/hateful_memes/configs/mmbt/defaults.yaml \
    model=mmbt \
    dataset=hateful_memes \
    run_type=test \
    checkpoint.resume_pretrained=False
```

This command will output where the generated predictions csv file is stored.

## Submission for Challenge

Next you can upload the generated csv file on DrivenData in their [submissions](https://www.drivendata.org/competitions/70/hateful-memes-phase-2/data//submissions/) page for Hateful Memes.

More details will be added once the challenge submission phase is live.


## Predicting for Phase 1

If you want to submit prediction for phase 1, you will need to use command line opts to override the jsonl files that are loaded. At the end of any command you run, add the following to load seen dev and test splits:

```sh
dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_seen.jsonl \
dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_seen.jsonl
```

This will load the phase 1 files for you and evaluate those.

## Building on top of MMF and Open Sourcing your code

To understand how you build on top of MMF for your own custom models and then open source your code, take a look at this [example](https://github.com/apsdehal/hm_example_mmf).
