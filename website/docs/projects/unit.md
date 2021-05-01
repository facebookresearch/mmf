---
id: unit
sidebar_label: UniT
title: "UniT: Multimodal Multitask Learning with a Unified Transformer"
---

This project page shows how to use the UniT model from the following paper, released under the MMF:

- R. Hu, A. Singh. _UniT: Multimodal Multitask Learning with a Unified Transformer_. arXiv preprint arXiv:2102.10772, 2021 ([PDF](https://arxiv.org/pdf/2102.10772.pdf))

```
@article{hu2021unit,
  title={UniT: Multimodal multitask learning with a unified transformer},
  author={Hu, Ronghang and Singh, Amanpreet},
  journal={arXiv preprint arXiv:2102.10772},
  year={2021}
}
```

### Abstract

We propose UniT, a Unified Transformer model to simultaneously learn the most prominent tasks across different domains, ranging from object detection to natural language understanding and multimodal reasoning. Based on the transformer encoder-decoder architecture, our UniT model encodes each input modality with an encoder and makes predictions on each task with a shared decoder over the encoded input representations, followed by task-specific output heads. The entire model is jointly trained end-to-end with losses from each task. Compared to previous efforts on multi-task learning with transformers, we share the same model parameters across all tasks instead of separately fine-tuning task-specific models and handle a much higher variety of tasks across different domains. In our experiments, we learn 7 tasks jointly over 8 datasets, achieving strong performance on each task with 87.5% fewer parameters.

## Evaluating pretrained UniT models

:::tip

Install MMF following the [installation guide](https://mmf.sh/docs/getting_started/installation/).

:::

We provide a pretrained single UniT model (shared decoder, with COCO init.) on 8-datasets: COCO detection, Visual Genome (VG) detection, VQAv2, SNLI-VE, QNLI, MNLI-mm, QQP, SST-2. This model corresponds to Table 3 line 5 in [the paper](https://arxiv.org/pdf/2102.10772.pdf). You can evaluate this pretrained model as follows:

```bash
# evaluating using a single GPU on 8 datasets
CUDA_VISIBLE_DEVICES=0 python mmf_cli/run.py \
    config=projects/unit/configs/all_8_datasets/shared_dec.yaml \
    datasets=detection_coco,detection_visual_genome,vqa2,visual_entailment,glue_qnli,glue_sst2,glue_mnli_mismatched,glue_qqp \
    model=unit run_type=val training.batch_size=1 \
    checkpoint.resume_zoo=unit.all_8_datasets.shared_dec_with_coco_init
```
This command will download all the 8 datasets, the pretrained UniT model, and necessary prerequisites. It should give the following evaluation results on the 8 datasets:
```
- val/detection_coco/detection_mean_ap: 0.3899,
- val/detection_visual_genome/detection_mean_ap: 0.0329,
- val/vqa2/vqa_accuracy: 0.6696,
- val/visual_entailment/accuracy: 0.7316,
- val/glue_qnli/accuracy: 0.8790,
- val/glue_sst2/accuracy: 0.8922,
- val/glue_mnli_mismatched/accuracy: 0.8090,
- val/glue_qqp/accuracy: 0.9065
```

As an alternative, we also provide another pretrained model similar to the one above, but without using task embedding in the image and text encoders (so that it is slightly easier to extend this model to more downstream tasks), which can be run as follows.

```bash
# evaluating using a single GPU on 8 datasets
CUDA_VISIBLE_DEVICES=0 python mmf_cli/run.py \
    config=projects/unit/configs/all_8_datasets/shared_dec_without_task_embedding.yaml \
    datasets=detection_coco,detection_visual_genome,vqa2,visual_entailment,glue_qnli,glue_sst2,glue_mnli_mismatched,glue_qqp \
    model=unit run_type=val training.batch_size=1 \
    checkpoint.resume_zoo=unit.all_8_datasets.shared_dec_with_coco_init_without_task_embedding
```
It should give the following evaluation results on the 8 datasets:
```
- val/detection_coco/detection_mean_ap: 0.3847,
- val/detection_visual_genome/detection_mean_ap: 0.0331,
- val/vqa2/vqa_accuracy: 0.6825,
- val/visual_entailment/accuracy: 0.7407,
- val/glue_qnli/accuracy: 0.8815,
- val/glue_sst2/accuracy: 0.8850,
- val/glue_mnli_mismatched/accuracy: 0.8061,
- val/glue_qqp/accuracy: 0.9061
```

## Training your own models

:::tip

Please follow the [MMF documentation](https://mmf.sh/docs/getting_started/quickstart#training) for the training and evaluation of MMF models.

:::

Our models are trained using [Slurm](https://slurm.schedmd.com/quickstart.html). In our experiments, we use a batch size of 64 with 64 GPUs (with 8 nodes and 8 GPUs per node).

### Training the 8-dataset UniT model

We train the 8-dataset UniT model (shared decoder, with COCO init. Table 3 line 5 in [the paper](https://arxiv.org/pdf/2102.10772.pdf)) above as follows using Slurm (corresponding to row 10 in the table below)
```bash
# you may need to adapt the following line to your cluster setting
srun --mem=300g --nodes=8 --gres=gpu:8 --time=4300 --cpus-per-task=40 \
python mmf_cli/run.py \
    config=projects/unit/configs/all_8_datasets/shared_dec.yaml \
    datasets=detection_coco,detection_visual_genome,vqa2,visual_entailment,glue_qnli,glue_sst2,glue_mnli_mismatched,glue_qqp \
    model=unit run_type=train \
    env.save_dir=./save/unit/all_8_datasets/shared_dec \
    distributed.world_size=64 distributed.port=20000 \
    checkpoint.resume_zoo=unit.coco.single_task
```
This command will download all the 8 datasets and necessary prerequisites. Here `checkpoint.resume_zoo=unit.coco.single_task` means initializing from a single-task model trained on COCO (which we provide as part of the MMF model zoo). As an alternative, you also can train this COCO single-task model following the instruction below.

### Training other configurations

We also provide a set of configuration files for other datasets and settings in our experiments to facilitate the reproduction of our results. In the table below, we outline the different configurations:

| # | Datasets | Config Files (under `projects/unit/configs`) | Notes |
| --- | --- | --- | --- |
|  | single-task training |  |  |
| 1 | `detection_coco` | `coco/single_task.yaml` | paper Table 1 line 1 |
| 2 | `detection_visual_genome` | `vg/single_task.yaml` | paper Table 1 line 1 |
| 3 | `vqa2` | `vqa2/single_task.yaml` | paper Table 1 line 1 |
|  | COCO, VG, and VQAv2 |  |  |
| 4 | `detection_coco,` `vqa2` | `coco_vqa2/shared_dec.yaml` | paper Table 2 line 2** |
| 5 | `detection_coco,` `vqa2` | `coco_vqa2/separate_dec.yaml` | ** |
| 6 | `detection_visual_genome,` `vqa2` | `vg_vqa2/shared_dec.yaml` | paper Table 2 line 3** |
| 7 | `detection_visual_genome,` `vqa2` | `vg_vqa2/separate_dec.yaml` | ** |
| 8 | `detection_coco,` `detection_visual_genome,` `vqa2` | `coco_vg_vqa2/shared_dec.yaml` | paper Table 2 line 4** |
| 9 | `detection_coco,` `detection_visual_genome,` `vqa2` | `coco_vg_vqa2/separate_dec.yaml` | ** |
|  | all 8 datasets |  |  |
| 10 | `detection_coco,` `detection_visual_genome,` `vqa2,` `visual_entailment,` `glue_qnli,` `glue_sst2,` `glue_mnli_mismatched,` `glue_qqp` | `all_8_datasets/shared_dec.yaml` | paper Table 3 line 3 and 5** |
| 11 | `detection_coco,` `detection_visual_genome,` `vqa2,` `visual_entailment,` `glue_qnli,` `glue_sst2,` `glue_mnli_mismatched,` `glue_qqp` | `all_8_datasets/separate_dec.yaml` | paper Table 3 line 2 and 4** |
|  | without using task embedding |  |  |
| 12 | `detection_coco` | `coco/` `single_task_without_task_embedding.yaml` | No task embedding in encoders |
| 13 | `detection_coco,` `detection_visual_genome,` `vqa2,` `visual_entailment,` `glue_qnli,` `glue_sst2,` `glue_mnli_mismatched,` `glue_qqp` | `all_8_datasets/` `shared_dec_without_task_embedding.yaml` | No task embedding in encoders*** |

** To run with COCO init, add `checkpoint.resume_zoo=unit.coco.single_task` to the training command (alternatively, one can use `checkpoint.resume_file=xxx`, where `xxx` is the path to `unit_final.pth` in `env.save_dir` of the model trained in row 1)

*** To run with COCO init, add `checkpoint.resume_zoo=unit.coco.single_task_without_task_embedding` to the training command (alternatively, one can use `checkpoint.resume_file=xxx`, where `xxx` is the path to `unit_final.pth` in `env.save_dir` of the model trained in row 12)

For example:

1. to train a UniT model with Slurm:

```bash
# append checkpoint.resume_zoo as mentioned above to run with COCO init
# you may need to adapt the following line to your cluster setting
srun --mem=300g --nodes=8 --gres=gpu:8 --time=4300 --cpus-per-task=40 \
python mmf_cli/run.py \
    config=projects/unit/configs/<config_file> \
    datasets=<datasets> \
    model=unit run_type=train \
    env.save_dir=<specify_your_saving_dir> \
    distributed.world_size=64 distributed.port=20000
```

2. To evaluate a trained UniT model:

```bash
# evaluating using a single GPU on 8 datasets
CUDA_VISIBLE_DEVICES=0 python mmf_cli/run.py \
    config=projects/unit/configs/<config_file> \
    datasets=<datasets> \
    model=unit run_type=val training.batch_size=1 \
    env.save_dir=<same_saving_dir_as_in_training> \
    checkpoint.resume=True checkpoint.resume_best=True
```
