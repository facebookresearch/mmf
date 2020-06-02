# LoRRA

This repository contains the code for LoRRA model. Please cite the following paper if you are using LoRRA model from mmf:

* Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., ... & Rohrbach, M. (2019). *Towards vqa models that can read*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 8317-8326). ([arXiV](https://arxiv.org/abs/1904.08920))
```
@inproceedings{singh2019TowardsVM,
  title={Towards VQA Models That Can Read},
  author={Singh, Amanpreet and Natarajan, Vivek and Shah, Meet and Jiang, Yu and Chen, Xinlei and Batra, Dhruv and Parikh, Devi and Rohrbach, Marcus},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html)

## Training
To train LoRRA model on the TextVQA dataset, run the following command
```
mmf_run config=projects/lorra/configs/textvqa/defaults.yaml run_type=train_val dataset=textvqa model=lorra
```
