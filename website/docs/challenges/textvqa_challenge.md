---
id: textvqa_challenge
title: TextVQA Challenge
sidebar_label: TextVQA Challenge
---

TextVQA Challenge is available at [this link](https://textvqa.org/challenge/).

In MMF, we provide the starter code for various baseline models for this challenge. TextVQA dataset will also be automatically downloaded during first training.

In this tutorial, we provide steps for running training and evaluation with M4C model on TextVQA dataset and generating submission file for the challenge. The same steps can be used for other models.

## Installation

Follow the prerequisites for installation of mmf [here](https://mmf.sh/docs/getting_started/installation).

## Training and Evaluation

### Training

For running training on `train` set, run the following command:

```bash
mmf_run config=projects/m4c/configs/textvqa/defaults.yaml \
    datasets=textvqa \
    model=m4c \
    run_type=train
```

This will train the `m4c` model on the dataset and generate the checkpoints and best trained model (`m4c_final.pth`) will be stored in an experiment folder under the `./save` directory by default (unless `env.save_dir` is overriden).

### Evaluation

Next run evaluation on the validation `val` set:

```bash
mmf_run config=projects/m4c/configs/textvqa/defaults.yaml \
    datasets=textvqa \
    model=m4c \
    run_type=val \
    checkpoint.resume_file=<path_to_trained_model>
```

This will give you the performance of your model on the validation set. The metric will be TextVQA Accuracy.

## Predictions for Challenge

After we trained the model and evaluated on the validation set, we will generate the predictions on the `test` set which can be submitted to the Test Standard phase. The prediction file should contain the following:

- Question ID, `question_id`
- Answer, `answer`

```json
[
  {
    "question_id": "INT",
    "answer": "STRING"
  },
  {
    "question_id": "...",
    "answer": "..."
  }
]
```

With MMF you can directly generate the predictions in the required submission format with the following command:

```bash
mmf_predict config=projects/m4c/configs/textvqa/defaults.yaml \
    datasets=textvqa \
    model=m4c \
    run_type=test \
    checkpoint.resume_file=<path_to_trained_model>
```

This command will output where the generated predictions JSON file is stored.

## Submission for Challenge

Next you can upload the generated json file to EvalAI page for TextVQA [here](https://evalai.cloudcv.org/web/challenges/challenge-page/551/submission). Follow these steps:

```
> Go to https://evalai.cloudcv.org/web/challenges/challenge-page/551/overview
> Select Submit Tab
> Select Validation Phase
> Select the file by click Upload file
> Write a model name
> Upload
```

To check your results, you can go in 'My submissions' section and select 'Validation Phase' and click on 'Result file'.

Now, you can either edit the M4C model to create your own model on top of it or create your own model inside MMF to beat M4C in challenge.
