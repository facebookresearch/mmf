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

This repository contains the checkpoint for the pytorch implementation of the VILLA model, released originally under this ([repo](https://github.com/zhegan27/VILLA)). Please cite the following papers if you are using VILLA model from mmf:

* Gan, Z., Chen, Y. C., Li, L., Zhu, C., Cheng, Y., & Liu, J. (2020). *Large-scale adversarial training for vision-and-language representation learning.* arXiv preprint arXiv:2006.06195. ([arXiV](https://arxiv.org/abs/2006.06195))
```
@inproceedings{gan2020large,
  title={Large-Scale Adversarial Training for Vision-and-Language Representation Learning},
  author={Gan, Zhe and Chen, Yen-Chun and Li, Linjie and Zhu, Chen and Cheng, Yu and Liu, Jingjing},
  booktitle={NeurIPS},
  year={2020}
}
```

## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Training

UNITER uses image region features extracted by [BUTD](https://github.com/peteanderson80/bottom-up-attention).
These are different features than those extracted in MMF and used by default in our datasets.
Support for BUTD feature extraction through pytorch in MMF is in the works.
However this means that the UNITER and VILLA checkpoints which are pretrained on BUTD features,
do not work out of the box on image region features in MMF.
You can still finetune these checkpoints in MMF on the Faster RCNN features used in MMF datasets for comparable performance.
This is what is done by default.
Or you can download BUTD features for the dataset you're working with and change the dataset in MMF to use these.

To train a fresh UNITER model on the VQA2.0 dataset, run the following command
```
mmf_run config=projects/uniter/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=uniter
```

To finetune a pretrained UNITER model on the VQA2.0 dataset,
```
mmf_run config=projects/uniter/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=uniter checkpoint.resume_zoo=uniter.pretrained
```
The finetuning configs for VQA2 are from the UNITER base 4-gpu [configs](https://github.com/ChenRocks/UNITER/blob/master/config/train-vqa-base-4gpu.json). For an example finetuning config with smaller batch size consider using the ViLT VQA2 training configs, however this may yield slightly lower performance.

To finetune a pretrained [VILLA](https://arxiv.org/pdf/2006.06195.pdf) model on the VQA2.0 dataset,
```
mmf_run config=projects/uniter/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=uniter checkpoint.resume_zoo=villa.pretrained
```

To pretrain UNITER on the masked COCO dataset, run the following command
```
mmf_run config=projects/uniter/configs/masked_coco/defaults.yaml run_type=train_val dataset=masked_coco model=uniter
```

Based on the config used and `do_pretraining` defined in the config, the model can use the pretraining recipe described in the UNITER paper, or be finetuned on downstream tasks.
