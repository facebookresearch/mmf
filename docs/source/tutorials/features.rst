Features
========

Pythia features:

- **Model Zoo**: Reference implementations for state-of-the-art vision and language model including
  LoRRA_ (SoTA on VQA and TextVQA), Pythia_ model (VQA 2018 challenge winner), BAN and BUTD_.
- **Multi-Tasking**: Support for multi-tasking which allows training on
  multiple datasets together.
- **Datasets**: Includes support for various datasets built-in including VQA, VizWiz,
  TextVQA, VisualDialog, MS COCO Captioning.
- **Modules**: Provides implementations for many commonly used layers
  in vision and language domain
- **Distributed**: Support for distributed training based on DataParallel
  as well as DistributedDataParallel.
- **Unopinionated**: Unopinionated about the dataset and model implementations
  built on top of it.
- **Customization**: Custom losses, metrics, scheduling, optimizers, tensorboard;
  suits all your custom needs.

You can use Pythia to **bootstrap** for your next vision and language multimodal
research project.

Pythia can also act as **starter codebase** for challenges around vision and
language datasets (TextVQA challenge, VQA challenge).

.. _lorra: https://arxiv.org/abs/1904.08920
.. _pythia: https://arxiv.org/abs/1807.09956
.. _butd: https://arxiv.org/abs/1707.07998
