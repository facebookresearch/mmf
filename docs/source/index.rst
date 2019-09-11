.. pythia documentation master file, created by
   sphinx-quickstart on Tue Apr 23 10:42:55 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/facebookresearch/pythia

.. raw:: html

    <embed>
        <div>
        <div style="width: 50%; margin: 0 auto;">
            <a href="https://readthedocs.org/projects/learnpythia/">
                <img style="width: 100%" alt="Pythia" src="https://i.imgur.com/wPgp4N4.png"/>
            </a>
        </div>

        <div style="display: flex; align-items: center; justify-content: center;">

            <div style="padding-right: 5px">
                <a href="https://learnpythia.readthedocs.io/en/latest/?badge=latest">
                    <img alt="Documentation Status" src="https://readthedocs.org/projects/learnpythia/badge/?version=latest"/>
                </a>
            </div>
            <div style="padding-right: 5px">
                <a href="https://colab.research.google.com/drive/1Z9fsh10rFtgWe4uy8nvU4mQmqdokdIRR">
                    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/>
                </a>
            </div>
            <div style="padding-right: 5px">
                <a href="https://circleci.com/gh/facebookresearch/pythia">
                    <img alt="CircleCI" src="https://circleci.com/gh/facebookresearch/pythia.svg?style=svg"/>
                </a>
            </div>
        </div>
        <br/>
        <br/>
        </div>
    </embed>

Pythia's Documentation
==================================

Pythia is a modular framework for supercharging vision and language
research built on top of PyTorch.

Citation
========

If you use Pythia in your work, please cite:

```text
@inproceedings{Singh2019TowardsVM,
  title={Towards VQA Models That Can Read},
  author={Singh, Amanpreet and Natarajan, Vivek and Shah, Meet and Jiang, Yu and Chen, Xinlei and Batra, Dhruv and Parikh, Devi and Rohrbach, Marcus},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

and

```text
@inproceedings{singh2019pythia,
  title={Pythia-a platform for vision \& language research},
  author={Singh, Amanpreet and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  booktitle={SysML Workshop, NeurIPS},
  volume={2018},
  year={2019}
}
```

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/quickstart
   tutorials/concepts
   tutorials/features
   tutorials/dataset
   tutorials/pretrained_models
   tutorials/challenge

.. toctree::
   :maxdepth: 1
   :caption: Library

   common/registry
   common/sample
   models/base_model
   modules/losses
   modules/metrics
   datasets/base_dataset_builder
   datasets/base_dataset
   datasets/base_task
   datasets/processors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
