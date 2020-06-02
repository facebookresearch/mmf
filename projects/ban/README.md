# BAN

This repository contains the code for BAN model. Please cite the following paper if you are using BAN model from mmf:

* Kim, J. H., Jun, J., & Zhang, B. T. (2018). *Bilinear attention networks*. In Advances in Neural Information Processing Systems (pp. 1564-1574). ([arXiV](https://arxiv.org/abs/1805.07932))
```
@inproceedings{kim2018bilinear,
  title={Bilinear attention networks},
  author={Kim, Jin-Hwa and Jun, Jaehyun and Zhang, Byoung-Tak},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1564--1574},
  year={2018}
}
```

## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Training
To train BAN model on the VQA2 dataset, run the following command
```
mmf_run config=projects/ban/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=ban
```
