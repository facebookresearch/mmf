
# MMF

<div align="center">
  <a href="https://mmf.readthedocs.io/en/latest/">
  <img alt="Documentation Status" src="https://readthedocs.org/projects/mmf/badge/?version=latest"/>
  </a>
  <a href="https://circleci.com/gh/facebookresearch/mmf">
  <img alt="CircleCI" src="https://circleci.com/gh/facebookresearch/mmf.svg?style=svg"/>
  </a>
</div>

**NOTE**: MMF is still in beta mode and will replace Pythia framework.
To get the latest Pythia code which doesn't contain MMF changes, please use the following command:

```
git clone --branch v0.3 https://github.com/facebookresearch/mmf pythia
```

MMF is a modular framework for vision and language multimodal research. Built on top of PyTorch, it features:

- **Model Zoo**: Reference implementations for state-of-the-art vision and language model including
[LoRRA](https://arxiv.org/abs/1904.08920) (SoTA on VQA and TextVQA),
[Pythia](https://arxiv.org/abs/1807.09956) model (VQA 2018 challenge winner), [BAN](https://arxiv.org/abs/1805.07932) and [BUTD](https://arxiv.org/abs/1707.07998).
- **Multi-Tasking**: Support for multi-tasking which allows training on multiple dataset together.
- **Datasets**: Includes support for various datasets built-in including VQA, VizWiz, TextVQA, VisualDialog and COCO Captioning.
- **Modules**: Provides implementations for many commonly used layers in vision and language domain
- **Distributed**: Support for distributed training based on DataParallel as well as DistributedDataParallel.
- **Unopinionated**: Unopinionated about the dataset and model implementations built on top of it.
- **Customization**: Custom losses, metrics, scheduling, optimizers, tensorboard; suits all your custom needs.

You can use MMF to **_bootstrap_** for your next vision and language multimodal research project.

MMF can also act as **starter codebase** for challenges around vision and
language datasets (TextVQA challenge, VQA challenge)

![MMF Examples](https://i.imgur.com/BP8sYnk.jpg)

## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Documentation

Learn more about MMF [here](https://mmf.readthedocs.io/en/latest/).

## Citation

If you use MMF in your work, please cite:

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

MMF is licensed under BSD license available in [LICENSE](LICENSE) file
