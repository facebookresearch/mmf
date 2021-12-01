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

## The VinVL Dataset
The VinVL dataset is a dataset wrapper that augments an existing dataset within MMF.
The VinVL dataset doesn't contain new images or text, but introduces label and attribute tags as strings.
VinVL requires unique inputs for finetuning and pretraining unsupported by general datasets.
To enable this functionality on arbitrary datasets,the VinVL dataset contains a base dataset,
and returns an augmented version of samples from thebase dataset.

For example,
the VQA2 dataset may return a sample {image, text}
The VinVL dataset when asked for a sample, will return
{image, text', rand_caption, rand_label}
text' = text + labels
rand_caption = text from a random example
rand_label = obj detection labels text for a random example

The VinVL dataset assumes:
The sample returned by the base dataset contains a key "text" with string text.
There exists a label_map json file path in the dataset configfor a json obj containing idx_to_attribute and idx_to_labelmaps.
VinVL OD uses VG labels, and this map can be downloaded from https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json
The features_db points to features generated from the VinVLfeature extraction script,
consult the VinVL feature extraction tutorial for more details.

This is why for VinVL finetuning and pretraining you should use dataset=vinvl,
then specify your base dataset in your configs.
Here is an example from projects/vinvl/configs/vqa2/defaults.yaml.

```yaml
includes:
- ../vqa2/defaults.yaml

dataset_config:
  vinvl:
    base_dataset_name: vqa2
    label_map: /private/home/ryanjiang/winoground/pretrained_models/VG-SGG-dicts-vgoi6-clipped.json
    base_dataset: ${dataset_config.vqa2}
    processors:
      text_processor:
        type: vinvl_text_tokenizer
        params:
          mask_probability: 0
```
Where vqa2/defaults.yaml contains the feature paths pointing to VinVL features.


## Training

After extracting features and redirecting your dataset config,
to train VinVL model from scratch on the VQA2.0 dataset, run the following command
```bash
mmf_run config=projects/vinvl/configs/vqa2/defaults.yaml run_type=train dataset=vinvl model=vinvl
```

To finetune a pretrained VinVL model on the VQA2.0 dataset, run the following command
```bash
mmf_run config=projects/vinvl/configs/vqa2/defaults.yaml run_type=train dataset=vinvl model=vinvl checkpoint.resume_zoo=vinvl.pretrained
```
