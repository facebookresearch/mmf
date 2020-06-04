# Quickstart [![](https://circleci.com/gh/facebookresearch/mmf.svg?style=svg)](https://circleci.com/gh/facebookresearch/mmf)

**Authors**: Amanpreet Singh

In this quickstart, we are going to train [M4C](https://github.com/facebookresearch/mmf/tree/master/projects/m4c) model on TextVQA. Follow instructions at the bottom
to train other models in MMF.


## Installation

Install MMF following the [installation documentation](./installation).

## Getting Data

In MMF datasets and required files will be downloaded automatically when we run training next. For more details about custom datasets and other advanced setups for datasets check the [dataset documentation](../tutorials/dataset).

## Training

Now we can start training by running the following command:

```
mmf_run config=projects/m4c/configs/textvqa/defaults.yaml datasets=textvqa model=m4c run_type=train_val
```

## Inference

For running inference or generating predictions, we can specify a pretrained model using its zoo key and then run the following command:

```
mmf_predict config=projects/m4c/configs/textvqa/defaults.yaml datasets=textvqa model=m4c run_type=test checkpoint.resume_zoo=m4c.textvqa.defaults
```

For running inference on `val` set, use `run_type=val` and rest of the arguments remain same. Check more details in [pretrained models](pretrained_models) section.

These commands should be enough to get you started with training and performing inference using MMF.

## Citation

If you use MMF in your work, please cite:


```text
@inproceedings{singh2019pythia,
  title={Pythia-a platform for vision \& language research},
  author={Singh, Amanpreet and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  booktitle={SysML Workshop, NeurIPS},
  volume={2018},
  year={2019}
}
```

## Next steps

To dive deep into world of MMF, you can move on the following next topics:

- [Concepts and Terminology](../tutorials/concepts)
- [Using Pretrained Models](./pretrained_models)
- [Challenge Participation](./challenge)
- [FAQs](./faq)
