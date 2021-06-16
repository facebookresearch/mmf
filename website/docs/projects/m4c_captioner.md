---
id: m4c_captioner
sidebar_label: M4C-Captioner
title: 'TextCaps: a Dataset for Image Captioning with Reading Comprehension'
---

This project page shows how to use M4C-Captioner model from the following paper, released under the MMF:

- O. Sidorov, R. Hu, M. Rohrbach, A. Singh, _TextCaps: a Dataset for Image Captioning with Reading Comprehension_. in ECCV, 2020 ([PDF](https://arxiv.org/pdf/2003.12462.pdf))

```
@inproceedings{sidorov2019textcaps,
  title={TextCaps: a Dataset for Image Captioningwith Reading Comprehension},
  author={Sidorov, Oleksii and Hu, Ronghang and Rohrbach, Marcus and Singh, Amanpreet},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

Project Page: https://textvqa.org/textcaps

## Installation

Install MMF following the [installation guide](https://mmf.sh/docs/getting_started/installation/).

This will install all M4C dependencies such as `transformers` and `editdistance` and will also compile the python interface for PHOC features.

In addition, it is also necessary to install `pycocoevalcap`:
```
# install pycocoevalcap
# use the repo below instead of https://github.com/tylin/coco-caption
# note: you also need to have java on your machine
pip install git+https://github.com/ronghanghu/coco-caption.git@python23
```
**Note that java is required for [`pycocoevalcap`](https://github.com/ronghanghu/coco-caption)**.

## Pretrained M4C-Captioner Models

We release two variants of the M4C-Captioner model trained on the TextCaps dataset, one trained with newer features extracted with maskrcnn-benchmark (`defaults`), and the other trained with older features extracted with Caffe2 (`with_caffe2_feat`), which is used in our experimentations in the paper and has higher CIDEr. **Please use `with_caffe2_feat` config and model zoo file if you would like to exactly reproduce the results from our paper.**

| Config Files (under `projects/m4c_captioner/configs/m4c_captioner/textcaps`) | Pretrained Model Key | Metrics | Notes |
| --- | --- | --- | --- |
| `defaults.yaml` | `m4c_captioner.textcaps.defaults` | val CIDEr -- 89.1 (BLEU-4 -- 23.4) | newer features extracted with maskrcnn-benchmark |
| `with_caffe2_feat.yaml` | `m4c_captioner.textcaps.with_caffe2_feat` | val CIDEr -- 89.6 (BLEU-4 -- 23.3) | older features extracted with Caffe2; **used in experiments in the paper** |

## Training and Evaluating M4C-Captioner

Please follow the [MMF documentation](https://learnpythia.readthedocs.io/en/latest/tutorials/quickstart.html#training) for the training and evaluation of the M4C-Captioner models.

For example:

1) to train the M4C-Captioner model on the TextCaps training set:
```bash
mmf_run datasets=textcaps \
    model=m4c_captioner \
    config=projects/m4c_captioner/configs/m4c_captioner/textcaps/defaults.yaml \
    env.save_dir=./save/m4c_captioner/defaults \
    run_type=train_val
```

(Replace `projects/m4c_captioner/configs/m4c_captioner/textcaps/defaults.yaml` with other config files to train with other configurations. See the table above. You can also specify a different path to `env.save_dir` to save to a location you prefer.)

2) to generate prediction json files for the TextCaps (assuming you are evaluating the pretrained model `m4c_captioner.textcaps.defaults`):

Generate prediction file on the validation set:
```bash
mmf_predict datasets=textcaps \
    model=m4c_captioner \
    config=projects/m4c_captioner/configs/m4c_captioner/textcaps/defaults.yaml \
    env.save_dir=./save/m4c_captioner/defaults \
    run_type=val \
    checkpoint.resume_zoo=m4c_captioner.textcaps.defaults
```

Generate prediction file on the test set:
```bash
mmf_predict datasets=textcaps \
    model=m4c_captioner \
    config=projects/m4c_captioner/configs/m4c_captioner/textcaps/defaults.yaml \
    env.save_dir=./save/m4c_captioner/defaults \
    run_type=test \
    checkpoint.resume_zoo=m4c_captioner.textcaps.defaults
```

As with training, you can replace `config` and `checkpoint.resume_zoo` according to the setting you want to evaluate.

:::note

Use `checkpoint.resume=True` AND `checkpoint.resume_best=True` instead of `checkpoint.resume_zoo=m4c_captioner.textcaps.defaults` to evaluate your trained snapshots.

:::

:::tip

Follow [checkpointing](https://mmf.sh/docs/tutorials/checkpointing) tutorial to understand more fine-grained details of checkpoint, loading and resuming in MMF

:::

Afterwards, use `projects/m4c_captioner/scripts/textcaps_eval.py` to evaluate the prediction json file. For example:
```bash
# the default data location of MMF (unless you have specified it otherwise)
# this is where MMF datasets are stored
export MMF_DATA_DIR=~/.cache/torch/mmf/data

python projects/m4c_captioner/scripts/textcaps_eval.py \
    --set val \
    --annotation_file ${MMF_DATA_DIR}/datasets/textcaps/defaults/annotations/imdb_val.npy \
    --pred_file YOUR_VAL_PREDICTION_FILE
```
For test set evaluation, please submit to the TextCaps EvalAI server. See https://textvqa.org/textcaps for details.
