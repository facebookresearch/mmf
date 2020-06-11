.. mmf documentation master file, created by
   sphinx-quickstart on Tue Apr 23 10:42:55 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/facebookresearch/mmf

MMF
===

.. raw:: html

    <embed>
        <div style="display: flex; align-items: center; justify-content: center;">

            <div style="padding-right: 5px">
                <a href="https://mmf.readthedocs.io/en/latest/">
                    <img alt="Documentation Status" src="https://readthedocs.org/projects/mmf/badge/?version=latest"/>
                </a>
            </div>
            <div style="padding-right: 5px">
                <a href="https://circleci.com/gh/facebookresearch/mmf">
                    <img alt="CircleCI" src="https://circleci.com/gh/facebookresearch/mmf.svg?style=svg"/>
                </a>
            </div>
        </div>
        <br/>
        <br/>
        </div>
    </embed>


MMF is a modular framework for supercharging vision and language
research built on top of PyTorch. Using MMF, researchers and devlopers can train
custom models for VQA, Image Captioning, Visual Dialog, Hate Detection and other vision
and language tasks.

Read docs_ for tutorials and documentation.

Citation
========

If you use MMF in your work or use any models published in MMF, please cite:

.. code-block:: text

    @misc{singh2020mmf,
    author =       {Singh, Amanpreet and Goswami, Vedanuj and Natarajan, Vivek and Jiang, Yu and Chen, Xinlei and Shah, Meet and
                    Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
    title =        {MMF: A multimodal framework for vision and language research},
    howpublished = {\url{https://github.com/facebookresearch/mmf}},
    year =         {2020}
    }

.. toctree::
   :maxdepth: 1
   :caption: Library Reference

   lib/common/registry
   lib/common/sample
   lib/models/base_model
   lib/modules/losses
   lib/modules/metrics
   lib/datasets/base_dataset_builder
   lib/datasets/base_dataset
   lib/datasets/processors
   lib/utils/text



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _docs: https://mmf.sh/docs
