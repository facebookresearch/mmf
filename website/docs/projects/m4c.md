---
id: m4c
sidebar_label: M4C
title: Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA
---

This project page shows how to use M4C model from the following paper, released under the MMF:

- R. Hu, A. Singh, T. Darrell, M. Rohrbach, _Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA_. in CVPR, 2020 ([PDF](https://arxiv.org/pdf/1911.06258.pdf))

```
@inproceedings{hu2020iterative,
  title={Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA},
  author={Hu, Ronghang and Singh, Amanpreet and Darrell, Trevor and Rohrbach, Marcus},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

Project Page: http://ronghanghu.com/m4c

## Installation

Install MMF following the [installation guide](https://mmf.sh/docs/getting_started/installation/).

This will install all M4C dependencies such as `transformers` and `editdistance` and will also compile the python interface for PHOC features.

## Notes about data

This repo supports training and evaluation of the M4C model under three datasets: TextVQA, ST-VQA, and OCR-VQA. As you run a command, these datasets and the requirements would be automatically downloaded for you.

For the ST-VQA dataset, we notice that many images from COCO-Text in the downloaded ST-VQA data (around 1/3 of all images) are resized to 256×256 for unknown reasons, which degrades the image quality and distorts their aspect ratios. In the released object and OCR features below, we replaced these images with their original versions from COCO-Text as inputs to object detection and OCR systems.

The released imdbs contain OCR results and normalized bounding boxes (i.e. in the range of `[0,1]`) of each detected objects (under `obj_normalized_boxes` key) and OCR tokens (under `ocr_normalized_boxes` key). Note that the answers in ST-VQA and OCR-VQA imdbs are tiled (duplicated) to 10 answers per question to make its format consistent with the TextVQA imdbs.

For the TextVQA dataset, the downloaded file contains both imdbs with the Rosetta-en OCRs (better performance) and imdbs with Rosetta-ml OCRs (same OCR results as in the previous [LoRRA](http://openaccess.thecvf.com/content_CVPR_2019/papers/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.pdf) model). Please download the corresponding OCR feature files.

Note that the object Faster R-CNN features are extracted with [`extract_features_vmb.py`](https://github.com/facebookresearch/mmf/blob/master/tools/scripts/features/extract_features_vmb.py) and the OCR Faster R-CNN features are extracted with [`extract_ocr_frcn_feature.py`](https://github.com/facebookresearch/mmf/blob/master/projects/m4c/scripts/extract_ocr_frcn_feature.py).

## Pretrained M4C Models

We release the following pretrained models for M4C on three datasets: TextVQA, ST-VQA, and OCR-VQA.

For the TextVQA dataset, we release three versions: M4C trained with ST-VQA as additional data (our best model) with Rosetta-en, M4C trained on TextVQA alone with Rosetta-en, and M4C trained on TextVQA alone with Rosetta-ml (same OCR results as in the previous [LoRRA](http://openaccess.thecvf.com/content_CVPR_2019/papers/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.pdf) model).

| Datasets | Config Files (under `projects/m4c/configs`) | Pretrained Model Key | Metrics | Notes |
| --- | --- | --- | --- | --- |
| TextVQA (`textvqa`) | `textvqa/join_with_stvqa.yaml` | `m4c.textvqa.with_stvqa` | val accuracy - 40.55%; test accuracy - 40.46% | Rosetta-en OCRs; ST-VQA as additional data |
| TextVQA (`textvqa`) | `textvqa/defaults.yaml` | `m4c.textvqa.alone` | val accuracy - 39.40%; test accuracy - 39.01% | Rosetta-en OCRs |
| TextVQA (`textvqa`) | `textvqa/ocr_ml.yaml` | m4c.textvqa.ocr_ml | val accuracy - 37.06% | Rosetta-ml OCRs |
| ST-VQA (`stvqa`) | `stvqa/defaults.yaml` | `m4c.stvqa.defaults` | val ANLS - 0.472 (accuracy - 38.05%); test ANLS - 0.462 | Rosetta-en OCRs |
| OCR-VQA (`ocrvqa`) | `ocrvqa/defaults.yaml` | `m4c.ocrvqa.defaults` | val accuracy - 63.52%; test accuracy - 63.87% | Rosetta-en OCRs |

## Training and Evaluation

Please follow the [MMF documentation](https://mmf.sh/docs/getting_started/quickstart#training) for the training and evaluation of the M4C model on each dataset.

For example:

1. to train the M4C model on the TextVQA training set:

```bash
mmf_run dataset=textvqa \
  model=m4c \
  config=projects/m4c/configs/textvqa/defaults.yaml \
  env.save_dir=./save/m4c
```

(Replace `textvqa` with other datasets and `projects/m4c/configs/textvqa/defaults.yaml` with other config files to train with other datasets and configurations. See the table above. You can also specify a different path to `env.save_dir` to save to a location you prefer.)

2. To evaluate the pretrained M4C model locally on the a TextVQA's validation set (assuming that the pretrained model that you are evaluating is `m4c.textvqa.with_stvqa`):

```bash
mmf_run dataset=textvqa \
  model=m4c \
  config=projects/m4c/configs/textvqa/defaults.yaml \
  env.save_dir=./save/m4c \
  run_type=val \
  checkpoint.resume_zoo=m4c.textvqa.with_stvqa
```

As with training, you can replace `dataset`, `config` and `checkpoint.resume_zoo` according to the setting you want to evaluate.

:::note

Use `checkpoint.resume=True` AND `checkpoint.resume_best=True` instead of `checkpoint.resume_zoo=m4c.textvqa.with_stvqa` to evaluate your trained snapshots.

:::

:::tip

Follow [checkpointing](https://mmf.sh/docs/tutorials/checkpointing) tutorial to understand more fine-grained details of checkpoint, loading and resuming in MMF

:::

3. to generate the EvalAI prediction files for the TextVQA test set (assuming you are evaluating the pretrained model `m4c.textvqa.with_stvqa`):

```bash
mmf_predict dataset=textvqa \
  model=m4c \
  config=projects/m4c/configs/textvqa/defaults.yaml \
  env.save_dir=./save/m4c \
  run_type=test \
  checkpoint.resume_zoo=m4c.textvqa.with_stvqa
```

As before, for generating prediction for other pretrained model for TextVQA, replace `config` and `checkpoint.resume_zoo` according to the setting you want in the table.

:::note

To generate predictions on val set, use `run_type=val` instead of `run_type=test`. As before, to generate predictions for your checkpoint, use `checkpoint.resume=True` AND `checkpoint.resume_best=True` instead of `checkpoint.resume_zoo=m4c.textvqa.with_stvqa`.

:::
