# Hateful Memes

The Hateful Memes challenge is available at [this link](https://www.drivendata.org/competitions/64/hateful-memes).

In MMF, we provide the starter code and baseline pretrained models for this challenge and the configurations used for training the reported baselines. For more details check [this link](https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes).

In this tutorial, we provide steps for running training and evaluation with MMBT model on hateful memes dataset and generating submission file for the challenge. The same steps can be used for your own models.

## Installation and Preparing the dataset

Follow the prerequisites for installation and dataset [here](https://github.com/facebookresearch/mmf/tree/master/projects/hateful_memes#prerequisites).

## Training and Evaluation

### Training
For running training on train set, run the following command:
```
mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml model=mmbt dataset=hateful_memes training.run_type=train_val
```
This will train the `mmbt` model on the dataset and generate the checkpoints and best trained model (`mmbt_final.pth`) will be stored in the `./save` directory by default.

### Evaluation

Next run evaluation on the validation set:
```
mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml model=mmbt dataset=hateful_memes training.run_type=val resume_file=./save/mmbt_final.pth
```
This will give you the performance of your model on the validation set. The metrics are AUROC, ACC, Binary F1 etc.


## Predictions for Challenge

After we trained the model and evaluated on the validation set, we will generate the predictions on the test set. The prediction file should contain the following three columns:

- Meme identification number, `id`
- Probability that the meme is hateful, `proba`
- Binary label that the meme is hateful (1) or non-hateful (0), `label`


With MMF you can directly generate the predictions in the required submission format with the following command:

```
mmf_predict config=projects/hateful_memes/configs/mmbt/defaults.yaml model=mmbt dataset=hateful_memes run_type=test
```

This command will output where the generated predictions csv file is stored.

## Submission for Challenge

Next you can upload the generated csv file on DrivenData in their [submissions](https://www.drivendata.org/competitions/64/hateful-memes/submissions/) page for Hateful Memes.

More details will be added once the challenge submission phase is live.
