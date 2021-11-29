---
id: vinvl
sidebar_label: VinVL
title: "VinVL: Revisiting Visual Representations in Vision-Language Models"
---

This repository contains the code for pytorch implementation of VinVL model, released originally under this ([repo](https://github.com/microsoft/Oscar)). Please cite the following papers if you are using VinVL model from mmf:

* Zhang, P., Li, X., Hu, X., Yang, J., Zhang, L., Wang, L., ... & Gao, J. (2021). *Vinvl: Revisiting visual representations in vision-language models*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5579-5588). ([arXiV](https://arxiv.org/abs/2101.00529))
```
@article{li2020oscar,
  title={Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks},
  author={Li, Xiujun and Yin, Xi and Li, Chunyuan and Hu, Xiaowei and Zhang, Pengchuan and Zhang, Lei and Wang, Lijuan and Hu, Houdong and Dong, Li and Wei, Furu and Choi, Yejin and Gao, Jianfeng},
  journal={ECCV 2020},
  year={2020}
}

@article{zhang2021vinvl,
  title={VinVL: Making Visual Representations Matter in Vision-Language Models},
  author={Zhang, Pengchuan and Li, Xiujun and Hu, Xiaowei and Yang, Jianwei and Zhang, Lei and Wang, Lijuan and Choi, Yejin and Gao, Jianfeng},
  journal={CVPR 2021},
  year={2021}
}
```

## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Features

VinVL's main contribution was in showing the impact of visual representations in image region features on VL models.
To use their image features consider either downloaded pre-extracted features from VinVL and remapping them into a MMF
dataset. Or running parallel VinVL feature extraction on an image directory using the MMF VinVL feature extraction script under
`mmf/tools/scripts/features/extract_features_vinvl.py`. After extracting these features to an output directory,
change the feature paths in the dataset config in `mmf/configs/datasets/<your dataset name>/defaults.yaml`
to point to your new features.


## Training

After extracting features and redirecting your dataset config,
to train VinVL model from scratch on the VQA2.0 dataset, run the following command
```
mmf_run config=projects/vinvl/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=vinvl
```

To finetune a pretrained VinVL model on the VQA2.0 dataset, run the following command
```
mmf_run config=projects/vinvl/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=vinvl checkpoint.resume_zoo=vinvl.pretrained
```
