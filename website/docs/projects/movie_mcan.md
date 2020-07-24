---
id: movie_mcan
sidebar_label: Movie+MCAN (VQA 2020 Winner)
title: MoViE+MCAN (VQA 2020 Challenge Winner)
---

This is a tutorial for running the MoViE+MCAN model which won the VQA Challenge at CVPR 2020. The winning team comprised of Nguyen, D. K., Jiang, H., Goswami, V., Yu. L. & Chen, X. MoViE+MCAN model is derived from the following papers, and is released under the MMF. Please cite both these papers if you use the model or the grid features used to train this model in your work:

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

Install MMF following the [installation guide](https://mmf.sh/docs/getting_started/installation/).

## Data Setup

Annotations and features for VQA2.0 and VisualGenome will be automatically downloaded. The grid image features were extracted using the models trained in this [repo](https://github.com/facebookresearch/grid-feats-vqa). Other variants of features data available in that [repo](https://github.com/facebookresearch/grid-feats-vqa) can also be used.

## Training and Evaluation

To train MoViE+MCAN model on the VQA2.0 + Visual Genome dataset, run:

```bash
mmf_run config=projects/movie_mcan/configs/vqa2/defaults.yaml \
    model=movie_mcan \
    dataset=vqa2 \
    run_type=train
```

this will save the trained model `movie_mcan_final.pth` in your `./save` directory for the experiment.

To evaluate the trained model on the VQA2.0 val set, run:

```bash
mmf_run config=projects/movie_mcan/configs/vqa2/defaults.yaml \
    model=movie_mcan \
    dataset=vqa2 \
    run_type=val \
    checkpoint.resume_file=<path_to_trained_pth_file>
```

## Inference Prediction for Eval AI Submission

To generate the vqa prediction file for Eval AI submission on `test-dev`, run:

```bash
mmf_predict config=projects/movie_mcan/configs/vqa2/defaults.yaml \
    model=movie_mcan \
    dataset=vqa2 \
    run_type=test \
    checkpoint.resume_file=<path_to_trained_pth_file>
```

## Pretrained model

| Datasets | Config File | Pretrained Model Key | Metrics | Notes |
| --- | --- | --- | --- | --- |
| VQA2.0 (`vqa2`) | `projects/movie_mcan/configs/vqa2/defaults.yaml` | `movie_mcan.grid.vqa2_vg` | testdev accuracy - 73.92% | Uses Visual Genome as extra data |


To generate predictions with the pretrained MoViE+MCAN model on VQA2.0 `test-dev` set (assuming that the pretrained model that you are evaluating is `movie_mcan.grid.vqa2_vg`), run:

```bash
mmf_predict config=projects/movie_mcan/configs/vqa2/defaults.yaml \
  dataset=vqa2 \
  model=movie_mcan \
  run_type=test \
  checkpoint.resume_zoo=movie_mcan.grid.vqa2_vg
```

:::tip

Follow [checkpointing](https://mmf.sh/docs/tutorials/checkpointing) tutorial to understand more fine-grained details of checkpoint, loading and resuming in MMF

:::
