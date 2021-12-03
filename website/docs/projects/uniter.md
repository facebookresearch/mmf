---
id: uniter
sidebar_label: UNITER
title: "UNITER: UNiversal Image-TExt Representation Learning"
---

This repository contains the code for pytorch implementation of UNITER model, released originally under this ([repo](https://github.com/ChenRocks/UNITER)). Please cite the following papers if you are using UNITER model from mmf:

* Chen, Y.-C., Li, L., Yu, L., Kholy, A. E., Ahmed, F., Gan,
Z., Cheng, Y., and jing Liu, J. *Uniter: Universal imagetext representation learning.* In European Conference on
Computer Vision, 2020b. ([arXiV](https://arxiv.org/pdf/1909.11740))
```
@inproceedings{chen2020uniter,
  title={Uniter: Universal image-text representation learning},
  author={Chen, Yen-Chun and Li, Linjie and Yu, Licheng and Kholy, Ahmed El and Ahmed, Faisal and Gan, Zhe and Cheng, Yu and Liu, Jingjing},
  booktitle={ECCV},
  year={2020}
}
```

## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Training

To train a fresh UNITER model on the VQA2.0 dataset, run the following command
```
mmf_run config=projects/uniter/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=uniter
```

To finetune a pretrained UNITER model on the VQA2.0 dataset,
```
mmf_run config=projects/uniter/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=uniter checkpoint.resume_zoo=uniter.pretrained
```

To finetune a pretrained [VILLA](https://arxiv.org/pdf/2006.06195.pdf) model on the VQA2.0 dataset,
```
mmf_run config=projects/uniter/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=uniter checkpoint.resume_zoo=villa.pretrained
```

To pretrain UNITER on the masked COCO dataset, run the following command
```
mmf_run config=projects/uniter/configs/masked_coco/defaults.yaml run_type=train_val dataset=masked_coco model=uniter
```


Based on the config used and `do_pretraining` defined in the config, the model can use the pretraining recipe described in the UNITER paper, or be finetuned on downstream tasks.
