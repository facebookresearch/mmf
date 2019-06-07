## Cycle-Consistency for Robust Visual Question Answering


### Training

* To train a cycle consistent model, follow download instructions as [here](https://github.com/facebookresearch/pythia/tree/0.1#quick-start)
* Download larger answer space and move it in the `data` folder from [here](https://dl.fbaipublicfiles.com/pythia/data/answers_vqa_larger.txt)
* Run the train script as follows:

To train with cycle consistent training scheme

```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/cycle_consistency/pythia_cycle_consistent.yml
```


To train with cycle consistent training scheme and failure prediction

```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/cycle_consistency/pythia_cycle_consistent_with_failure_prediction.yml
```


### Testing


```
CUDA_VISIBLE_DEVICES=0 python run_test.py --config config/cycle_consistency/pythia_cycle_consistent.yml \
                                          --model_path <path_to_downloaded_model>
```

### Pretrained Models and Predictions

| Description | performance (test-dev) | Link |
| --- | --- | --- |
| Pythia with Cycle Consistency | 68.87 | https://dl.fbaipublicfiles.com/pythia/pretrained_models/cycle_consistent.tar.gz |



### Citation

```
@inproceedings{meet2019cycle,
title={Cycle-Consistency for Robust Visual Question Answering},
author={Shah, Meet and Chen, Xinlei and Rohrbach, Marcus and Parikh, Devi},
booktitle={2019 Conference on Computer Vision and Pattern Recognition (CVPR)},
organization={IEEE}
}
```

For any question and improvements about the code, please contact Meet Shah (meetshah1995@gmail.com)
