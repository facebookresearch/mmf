<div align="center">
  <a href="https://readthedocs.org/projects/learnpythia/">
    <img width="60%" alt="Pythia" src="https://i.imgur.com/wPgp4N4.png"/>
  </a>
</div>

<div align="center">
  <a href="https://learnpythia.readthedocs.io/en/latest/?badge=latest">
  <img alt="Documentation Status" src="https://img.shields.io/readthedocs/pythia.svg?label=docs&style=flat&logo=read-the-docs"/>
  </a>
  <a href="https://colab.research.google.com/drive/1Z9fsh10rFtgWe4uy8nvU4mQmqdokdIRR">
  <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/>
  </a>
  <a href="https://circleci.com/gh/facebookresearch/pythia">
  <img alt="CircleCI" src="https://circleci.com/gh/facebookresearch/pythia.svg?style=svg"/>
  </a>
</div>

Pythia is a modular framework for vision and language multimodal research. Built on top
of PyTorch, it features:

- **Model Zoo**: Reference implementations for state-of-the-art vision and language model including
[LoRRA](https://arxiv.org/abs/1904.08920) (SoTA on VQA and TextVQA),
[Pythia](https://arxiv.org/abs/1807.09956) model (VQA 2018 challenge winner) , [BAN](https://arxiv.org/abs/1805.07932) and [BUTD](https://arxiv.org/abs/1707.07998).
- **Multi-Tasking**: Support for multi-tasking which allows training on multiple dataset together.
- **Datasets**: Includes support for various datasets built-in including VQA, VizWiz, TextVQA, VisualDialog and COCO Captioning.
- **Modules**: Provides implementations for many commonly used layers in vision and language domain
- **Distributed**: Support for distributed training based on DataParallel as well as DistributedDataParallel.
- **Unopinionated**: Unopinionated about the dataset and model implementations built on top of it.
- **Customization**: Custom losses, metrics, scheduling, optimizers, tensorboard; suits all your custom needs.

You can use Pythia to **_bootstrap_** for your next vision and language multimodal research project.

Pythia can also act as **starter codebase** for challenges around vision and
language datasets (TextVQA challenge, VQA challenge)

![Pythia Examples](https://i.imgur.com/BP8sYnk.jpg)

## Demo

1. [Pythia VQA](https://colab.research.google.com/drive/1Z9fsh10rFtgWe4uy8nvU4mQmqdokdIRR).
2. [BUTD Captioning](https://colab.research.google.com/drive/1vzrxDYB0vxtuUy8KCaGxm--nDCJvyBSg).

## Documentation

Learn more about Pythia [here](https://learnpythia.readthedocs.io/en/latest/).

## Citation

If you use Pythia in your work, please cite:

```
@inproceedings{singh2019TowardsVM,
  title={Towards VQA Models That Can Read},
  author={Singh, Amanpreet and Natarajan, Vivek and Shah, Meet and Jiang, Yu and Chen, Xinlei and Batra, Dhruv and Parikh, Devi and Rohrbach, Marcus},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

and

```
@inproceedings{singh2018pythia,
  title={Pythia-a platform for vision \& language research},
  author={Singh, Amanpreet and Goswami, Vedanuj and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  booktitle={SysML Workshop, NeurIPS},
  volume={2018},
  year={2018}
}
```

## License

Pythia is licensed under BSD license available in [LICENSE](LICENSE) file
