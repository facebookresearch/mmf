# MoViE-MCAN (VQA 2020 Challenge Winner)

This repository contains the code for pytorch implementation of MoViE+MCAN model, winner of the VQA Challenge 2020 at CVPR. Please cite the following papers if you are using MoViE+MCAN model from mmf:

* Nguyen, D. K., Goswami, V., & Chen, X. (2020). *Revisiting Modulated Convolutions for Visual Counting and Beyond*. arXiv preprint arXiv:2004.11883. ([arXiV](https://arxiv.org/abs/2004.11883))
```
@article{nguyen2020revisiting,
  title={Revisiting Modulated Convolutions for Visual Counting and Beyond},
  author={Nguyen, Duy-Kien and Goswami, Vedanuj and Chen, Xinlei},
  journal={arXiv preprint arXiv:2004.11883},
  year={2020}
}
```

and

* Jiang, H., Misra, I., Rohrbach, M., Learned-Miller, E., & Chen, X. (2020). *In Defense of Grid Features for Visual Question Answering*. arXiv preprint arXiv:2001.03615. ([arXiV](https://arxiv.org/abs/2001.03615))
```
@article{jiang2020defense,
  title={In Defense of Grid Features for Visual Question Answering},
  author={Jiang, Huaizu and Misra, Ishan and Rohrbach, Marcus and Learned-Miller, Erik and Chen, Xinlei},
  journal={arXiv preprint arXiv:2001.03615},
  year={2020}
}
```


## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Data Setup

[Data will be uploaded soon]

## Training

To train MoViE+MCAN model on the VQA2.0 dataset, run:

```
mmf_run config=projects/movie_mcan/configs/vqa2/defaults.yaml model=movie_mcan dataset=vqa2 run_type=train
```

## Validation

To validate the trained model on the VQA val set, run:

```
mmf_run config=projects/movie_mcan/configs/vqa2/defaults.yaml model=movie_mcan dataset=vqa2 run_type=val \
checkpoint.resume_file=<path_to_trained_pth_file>
```

## Inference Prediction for Eval AI Submission

To generate the vqa prediction file for Eval AI submission on test-dev, run:

```
mmf_predict config=projects/movie_mcan/configs/vqa2/defaults.yaml model=movie_mcan dataset=vqa2 run_type=test \
checkpoint.resume_file=<path_to_trained_pth_file>
```

## Pretrained model

[Pretrained model will be added soon]
