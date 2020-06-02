# ViLBERT

This repository contains the code for ViLBERT model, released originally under this ([repo](https://github.com/jiasenlu/vilbert_beta)). Please cite the following papers if you are using ViLBERT model from mmf:

* Lu, J., Batra, D., Parikh, D. and Lee, S., 2019. *Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks.* In Advances in Neural Information Processing Systems (pp. 13-23). ([arXiV](https://arxiv.org/abs/1908.02265))
```
@inproceedings{lu2019vilbert,
  title={Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={13--23},
  year={2019}
}
```

and

* Singh, A., Goswami, V., & Parikh, D. (2019). *Are we pretraining it right? Digging deeper into visio-linguistic pretraining*. arXiv preprint arXiv:2004.08744. ([arXiV](https://arxiv.org/abs/2004.08744))
```
@article{singh2020we,
  title={Are we pretraining it right? Digging deeper into visio-linguistic pretraining},
  author={Singh, Amanpreet and Goswami, Vedanuj and Parikh, Devi},
  journal={arXiv preprint arXiv:2004.08744},
  year={2020}
}
```

## Installation

Follow installation instructions in the [documentation](https://mmf.readthedocs.io/en/latest/notes/installation.html).

## Training
To train ViLBERT model on the VQA2.0 dataset, run the following command
```
mmf_run config=projects/vilbert/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=vilbert
```

Based on the config used and `training_head_type` defined in the config, the model can use either pretraining head or donwstream task specific heads(VQA, Vizwiz, SNLI-VE, MM IMDB or NLVR2).
