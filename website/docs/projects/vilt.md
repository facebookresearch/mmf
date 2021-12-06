---
id: vilt
sidebar_label: ViLT
title: "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision"
---

This repository contains the code for pytorch implementation of ViLT model, released originally under this ([repo](https://github.com/dandelin/ViLT)). Please cite the following papers if you are using ViLT model from mmf:

* Wonjae Kim, Bokyung Son, and Ildoo Kim. 2021. *ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision}*. In 38th International Conference on Machine Learning (ICML). ([arXiV](https://arxiv.org/pdf/2102.03334))
```
@misc{kim2021vilt,
      title={ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision},
      author={Wonjae Kim and Bokyung Son and Ildoo Kim},
      year={2021},
      eprint={2102.03334},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Training

To train ViLT model on the VQA2.0 dataset, run the following command
```
mmf_run config=projects/vilt/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=vilt
```

To finetune using different pretrained starting weights, change the `pretrained_model_name` under image_encoder in the config yaml to reference a huggingface model.
