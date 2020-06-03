# MMBT

This repository contains the code for pytorch implementation of MMBT model, released originally under this ([repo](https://github.com/facebookresearch/mmbt/)). Please cite the following paper if you are using MMBT model from mmf:

* Kiela, D., Bhooshan, S., Firooz, H., & Testuggine, D. (2019). *Supervised Multimodal Bitransformers for Classifying Images and Text.* arXiv preprint arXiv:1909.02950.. ([arXiV](https://arxiv.org/abs/1909.02950))
```
@article{kiela2019supervised,
  title={Supervised Multimodal Bitransformers for Classifying Images and Text},
  author={Kiela, Douwe and Bhooshan, Suvrat and Firooz, Hamed and Testuggine, Davide},
  journal={arXiv preprint arXiv:1909.02950},
  year={2019}
}
```


## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Training

To train MMBT model with grid features on the Hateful Memes dataset, run the following command
```
mmf_run config=projects/mmbt/configs/hateful_memes/defaults.yaml run_type=train_val dataset=hateful_memes model=mmbt
```

To train MMBT model with Faster RCNN region features on the Hateful Memes dataset, run the following command
```
mmf_run config=projects/mmbt/configs/hateful_memes/with_features.yaml run_type=train_val dataset=hateful_memes model=mmbt
```
