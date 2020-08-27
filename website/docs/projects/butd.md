---
id: butd
sidebar_label: BUTD
title: BUTD
---

This is a tutorial for running the BUTD model available in MMF. This model was released originally under this ([repo](https://github.com/peteanderson80/bottom-up-attention)). Please cite the following paper if you are using BUTD model from mmf:

* Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., & Zhang, L. (2018). *Bottom-up and top-down attention for image captioning and visual question answering*. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 6077-6086). ([arXiV](https://arxiv.org/abs/1707.07998))
```
@inproceedings{Anderson2017up-down,
  author = {Peter Anderson and Xiaodong He and Chris Buehler and Damien Teney and Mark Johnson and Stephen Gould and Lei Zhang},
  title = {Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering},
  booktitle={CVPR},
  year = {2018}
}
```


## Installation

Install MMF following the [installation guide](https://mmf.sh/docs/getting_started/installation/).

## Data Setup

For training the BUTD model on COCO captions we use the Karpathy splits. Annotations and features for COCO will be automatically downloaded.

## Training and Evaluation

To train BUTD model on the COCO karpathy train split, run:

```bash
mmf_run config=projects/butd/configs/coco/defaults.yaml \
    model=butd \
    dataset=coco \
    run_type=train
```

this will save the trained model `butd_final.pth` in your `./save` directory for the experiment.

To evaluate the trained model on the COCO val set, run:

```bash
mmf_run config=projects/butd/configs/coco/defaults.yaml \
    model=butd \
    dataset=coco \
    run_type=val \
    checkpoint.resume_file=<path_to_trained_pth_file>
```

BUTD evaluation can also be done with two other decoding variants with the same trained model, Beam Search and Nucleus Sampling. The following configs can be used :

- Beam Search Decoding (`projects/butd/configs/coco/beam_search.yaml`)
- Nucleus Sampling Decoding (`projects/butd/configs/coco/nucleus_sampling.yaml`)

```bash
mmf_run config=projects/butd/configs/coco/beam_search.yaml \
    model=butd \
    dataset=coco \
    run_type=val \
    checkpoint.resume_file=<path_to_trained_pth_file>
```


## Inference Prediction

To generate the coco captions prediction file for Karpathy `val` or `test` splits, run:

```bash
mmf_predict config=projects/butd/configs/coco/beam_search.yaml \
    model=butd \
    dataset=coco \
    run_type=val \
    checkpoint.resume_file=<path_to_trained_pth_file>
```

:::note

Evaluation predictions can only be generated using either `beam_search` or `nucleus_sampling` methods.

:::


## Pretrained model

| Datasets | Config File | Pretrained Model Key | Metrics |
| --- | --- | --- | --- | --- |
| COCO (`coco`) | `projects/butd/configs/coco/beam_search.yaml` | `butd` | val accuracy - 0.36 BLEU4 |


To generate predictions with the pretrained BUTD model on COCO Karpathy `val` set (assuming that the pretrained model that you are evaluating is `butd`), run:

```bash
mmf_predict config=projects/butd/configs/coco/beam_search.yaml \
    model=butd \
    dataset=coco \
    run_type=val \
    checkpoint.resume_zoo=butd
```

:::tip

Follow [checkpointing](https://mmf.sh/docs/tutorials/checkpointing) tutorial to understand more fine-grained details of checkpoint, loading and resuming in MMF

:::
