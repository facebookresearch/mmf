---
id: features
title: MMF Features
sidebar_label: MMF Features
---

MMF is powered by PyTorch and features:

- **Model Zoo**: Reference implementations for state-of-the-art vision and language model including [VisualBERT](https://arxiv.org/abs/1908.03557), [ViLBERT](https://arxiv.org/abs/1908.02265), [M4C](https://arxiv.org/abs/1911.06258) (SoTA on TextVQA/TextCaps), [Pythia](https://arxiv.org/abs/1807.09956) model (VQA 2018 challenge winner), and many others. See full list of projects within MMF [here](/docs/notes/projects).
- **Multi-Tasking**: Support for multi-tasking which allows training on multiple datasets together.
- **Datasets**: Includes support for various datasets built-in including VQA, VizWiz, TextVQA, VisualDialog and COCO Captioning. Running a single command automatically downloads and sets up the datasets for you.
- **Modules**: Provides implementations for many commonly used layers in vision and language domain.
- **Distributed**: Support for distributed training using DistributedDataParallel. With our hyperparameter sweep support, you can scale your model to any number of nodes.
- **Unopinionated**: Unopinionated about the dataset and model implementations built on top of it.
- **Customization**: Custom losses, metrics, scheduling, optimizers, tensorboard; suits all your custom needs.
