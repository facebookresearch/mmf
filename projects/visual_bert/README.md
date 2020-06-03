# VisualBERT

This repository contains the code for pytorch implementation of VisualBERT model, released originally under this ([repo](https://github.com/uclanlp/visualbert)). Please cite the following papers if you are using VisualBERT model from mmf:

* Li, L. H., Yatskar, M., Yin, D., Hsieh, C. J., & Chang, K. W. (2019). *Visualbert: A simple and performant baseline for vision and language*. arXiv preprint arXiv:1908.03557. ([arXiV](https://arxiv.org/abs/1908.03557))
```
@article{li2019visualbert,
  title={Visualbert: A simple and performant baseline for vision and language},
  author={Li, Liunian Harold and Yatskar, Mark and Yin, Da and Hsieh, Cho-Jui and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:1908.03557},
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

To train VisualBERT model on the VQA2.0 dataset, run the following command
```
mmf_run config=projects/visual_bert/configs/vqa2/defaults.yaml run_type=train_val dataset=vqa2 model=visual_bert
```

Based on the config used and `training_head_type` defined in the config, the model can use either pretraining head or donwstream task specific heads(VQA, Vizwiz, SNLI-VE, MM IMDB or NLVR2).
