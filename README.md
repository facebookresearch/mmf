# Pythia

Pythia is a modular framework for vision and language multimodal research. Built on top
of PyTorch, pythia features:

- Model zoo for reference implementations for state-of-the-art vision and language model including
[LoRRA](https://arxiv.org/abs/1904.08920) (SoTA on VQA and TextVQA),
[Pythia](https://arxiv.org/abs/1807.09956) model (VQA 2018 challenge winner) and [BAN]().
- Support for multi-tasking which allows training on multiple dataset together.
- Includes support for various datasets built-in including VQA, VizWiz, TextVQA and VisualDialog.
- Provides implementations for many commonly used layers in vision and language domain
- Support for distributed training based on DataParallel as well as DistributedDataParallel.
- Unopinionated about the dataset and model implementations built on top of it.
- Custom losses, metrics, scheduling, optimizers, tensorboard; suits all your custom needs.

You can use Pythia to bootstrap for your next vision and language multimodal research project.
Pythia can also act as starter codebase for challenges around vision and language datasets.

## Getting Started
