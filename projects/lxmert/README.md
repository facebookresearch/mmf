# LXMERT

This repository contains the code LXMERT  model, released originally under this ([repo](https://github.com/airsplay/lxmert)). Please cite the following paper if you are using LXMERT  model from mmf:

* Hao Tan and Mohit Bansal, 2019. *LXMERT: Learning Cross-Modality Encoder Representations from Transformers*. In Empirical Methods in Natural Language Processing. ([arXiV](https://arxiv.org/abs/1908.07490))
```
@inproceedings{tan2019lxmert,
  title={LXMERT: Learning Cross-Modality Encoder Representations from Transformers},
  author={Tan, Hao and Bansal, Mohit},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019}
}
```

## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Pretraining

To pretrain LXMERT model on the VQA2.0 dataset, run the following command

```
mmf_run config=projects/lxmert/configs/vqa2/pretrain.yaml run_type=train_val dataset=masked_vqa2 model=lxmert
```

to pretrain LXMERT model on the VQA2.0, COCO, and GQA datasets, run the following command:


```
mmf_run config=projects/lxmert/configs/pretrain.yaml run_type=train_val datasets=masked_vqa2,masked_gqa,masked_coco,visual_genome model=lxmert
```

## Finetuning

To finetune LXMERT model on the VQA2.0 dataset, run the following command

```
mmf_run config=projects/lxmert/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=lxmert
```

Based on the config used and `training_head_type` defined in the config, the model can use either pretraining head or donwstream task specific heads(VQA).
