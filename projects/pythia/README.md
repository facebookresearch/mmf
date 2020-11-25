# Pythia

This repository contains the code for pytorch implementation of Pythia v0.3 model. Please cite the following papers if you are using Pythia model:

The paper which introduced Pythia v0.3:

* Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D. and Rohrbach, M. (2019). *Towards vqa models that can read*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 8317-8326). ([arXiV](https://arxiv.org/abs/1904.08920))

```
@inproceedings{singh2019TowardsVM,
  title={Towards VQA Models That Can Read},
  author={Singh, Amanpreet and Natarajan, Vivek and Shah, Meet and Jiang, Yu and Chen, Xinlei and Batra, Dhruv and Parikh, Devi and Rohrbach, Marcus},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

and the original paper for old Pythia model:

* Jiang, Y., Natarajan, V., Chen, X., Rohrbach, M., Batra, D., & Parikh, D. (2018). *Pythia v0. 1: the winning entry to the vqa challenge 2018*. arXiv preprint arXiv:1807.09956. ([arXiV](https://arxiv.org/abs/1807.09956))
```
@article{jiang2018pythia,
  title={Pythia v0. 1: the winning entry to the vqa challenge 2018},
  author={Jiang, Yu and Natarajan, Vivek and Chen, Xinlei and Rohrbach, Marcus and Batra, Dhruv and Parikh, Devi},
  journal={arXiv preprint arXiv:1807.09956},
  year={2018}
}
```

## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Training
To train Pythia model on the VQA2.0 dataset, run the following command
```
mmf_run config=projects/pythia/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=pythia
```
