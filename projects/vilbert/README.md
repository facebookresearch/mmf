# ViLBERT

This repository contains the code for ViLBERT model, released originally under this ([repo](https://github.com/jiasenlu/vilbert_beta)). Please cite the following paper if you are using ViLBERT model from mmf:

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

## Installation

Clone this repository, and build it with the following command.
```
cd ~/mmf
python setup.py build develop
```

## Training
To train ViLBERT model on the VQA2.0 dataset, run the following command
```
python tools/run.py config=projects/vilbert/configs/vqa2/defaults.yaml training.run_type=train_val dataset=vqa2 model=vilbert
```

Based on the config used and `training_head_type` defined in the config, the model can use either pretraining head or donwstream task specific heads(VQA, Vizwiz, SNLI-VE, MM IMDB or NLVR2).
