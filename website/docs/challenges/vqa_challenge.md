---
id: vqa_challenge
title: VQA Challenge
sidebar_label: VQA Challenge
---

VQA Challenge is available at [this link](https://visualqa.org/challenge.html).

In MMF, we provide the starter code for various baseline models for this challenge. VQA2.0 dataset will also be automatically downloaded during first training.

In this tutorial, we provide steps for running training and evaluation with VisualBERT model on VQA2.0 dataset and generating submission file for the challenge. The same steps can be used for other models.

## Installation

Follow the prerequisites for installation of mmf [here](https://mmf.sh/docs/getting_started/installation).

## Training and Evaluation

### Training

For running training on train set, run the following command:

```bash
mmf_run config=projects/visual_bert/configs/vqa2/defaults.yaml \
    model=visual_bert \
    dataset=vqa2 \
    run_type=train
```

This will train the `visual_bert` model on the dataset and generate the checkpoints and best trained model (`visual_bert_final.pth`) will be stored in an experiment folder under the `./save` directory by default.

### Evaluation

Next run evaluation on the validation set:

```bash
mmf_run config=projects/visual_bert/configs/vqa2/defaults.yaml \
    model=visual_bert \
    dataset=vqa2 \
    run_type=val \
    checkpoint.resume_file=<path_to_trained_model>
```

This will give you the performance of your model on the validation set. The metric will be VQA Accuracy.

## Predictions for Challenge

After training the model and evaluated on the validation set, we will generate the predictions on the `test-dev` and `test-std` set. The prediction file should contain the following for each sample:

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
mmf_predict config=projects/visual_bert/configs/vqa2/defaults.yaml \
    model=visual_bert \
    dataset=vqa2 \
    run_type=test \
    checkpoint.resume_file=<path_to_trained_model>
```

This command will output where the generated predictions JSON file is stored.

## Submission for Challenge

Next you can upload the generated json file to EvalAI page for VQA [here](https://evalai.cloudcv.org/web/challenges/challenge-page/514/my-submission). To check your results, you can go in 'My submissions' section and check the phase where you submitted your results file.
