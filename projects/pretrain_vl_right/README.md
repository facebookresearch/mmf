# Visio-Linguistic Pretraining

This repository contains the code for modified implementation of VisualBERT and ViLBERT used in the folowwing paper. Please cite this paper if you are using these models:

* Singh, A., Goswami, V., & Parikh, D. (2019). *Are we pretraining it right? Digging deeper into visio-linguistic pretraining*.

TODO: Update citation bibtex once uplaoded to ArXiv.
```
```

## Installation

Clone this repository, and build it with the following command.
```
cd ~/pythia
python setup.py build develop
```

## Getting Data

We provide the different dataset features and ImDB files used for training the VisualBERT and ViLBERT models. Please follow the [Pythia documentation](https://learnpythia.readthedocs.io/en/latest/tutorials/quickstart.html#getting-data) to get data for each dataset as well as vocabulary files.

Below are the download links to the imdbs and features used for this project. **Note that LMDBs should be downloaded under `data/datasets/<dataset_name>/features/lmdbs/`. ImDB files should be extracted under `data/datasets/<dataset_name>/imdbs/`.**


| Datasets      | ImDBs | Feature LMDBs |
|--------------|----------|-------------------------------|
| COCO Captions     | [COCO ImDB](https://dl.fbaipublicfiles.com/pythia/data/datasets/coco/imdbs/coco_captions.tar.gz) | [COCO trainval](https://dl.fbaipublicfiles.com/pythia/data/datasets/coco/features/lmdbs/coco_trainval2014.lmdb) [COCO test](https://dl.fbaipublicfiles.com/pythia/data/datasets/coco/features/lmdbs/coco_test2015.lmdb) |
| VQA2.0      | [VQA2.0 ImDB](https://dl.fbaipublicfiles.com/pythia/data/datasets/vqa2/imdbs/vqa2.tar.gz) | [VQA2.0 trainval](https://dl.fbaipublicfiles.com/pythia/data/datasets/coco/features/lmdbs/coco_trainval2014.lmdb) [VQA2.0 test](https://dl.fbaipublicfiles.com/pythia/data/datasets/coco/features/lmdbs/coco_test2015.lmdb) |
| Concceptual Captions      | [CC ImDB](https://dl.fbaipublicfiles.com/pythia/data/datasets/cc/imdbs/cc_captions.tar.gz) | [CC train](https://dl.fbaipublicfiles.com/pythia/data/datasets/cc/features/lmdbs/cc_train.lmdb) [CC val](https://dl.fbaipublicfiles.com/pythia/data/datasets/cc/features/lmdbs/cc_val.lmdb) |
| Vizwiz      | [Vizwiz ImDB](https://dl.fbaipublicfiles.com/pythia/data/datasets/vizwiz/imdbs/vizwiz.tar.gz) | [Vizwiz](https://dl.fbaipublicfiles.com/pythia/data/datasets/vizwiz/features/lmdbs/vizwiz.lmdb) |
| SNLI-VE      | [SNLI ImDB](https://dl.fbaipublicfiles.com/pythia/data/datasets/visual_entailment/imdbs/visual_entailment.tar.gz) | [SNLI](https://dl.fbaipublicfiles.com/pythia/data/datasets/visual_entailment/features/lmdbs/flickr30k.lmdb) |
| MM-IMDB      | [MMIMDB ImDB](https://dl.fbaipublicfiles.com/pythia/data/datasets/mmimdb/imdbs/mmimdb.tar.gz) | [MMIMDB](https://dl.fbaipublicfiles.com/pythia/data/datasets/mmimdb/features/lmdbs/mmimdb.lmdb) |




## Training

### Pretraining

Example : To pretrain VisualBERT model on the COCO Captions dataset, run the following command
```
python tools/run.py config=projects/visual_bert/configs/masked_coco/pretrain.yaml training.run_type=train_val dataset=masked_coco model=visual_bert
```

### Finetuning

Example : To finetune VisualBERT model on the VQA2.0 dataset, run the following command
```
python tools/run.py config=projects/visual_bert/configs/vqa2/defaults.yaml training.run_type=train_val dataset=vqa2 model=visual_bert training.resume_file=<path_to_pretrained_visual_bert_model> training.load_pretrained=true
```

Configs for different settings and pretraining datasets are provided in the next section.

## Configs for different pretraining datasets:

#### VisualBERT Masked COCO

- [Masked COCO 100%](projects/visual_bert/configs/masked_coco/pretrain.yaml)
- [Masked COCO 50%](projects/pretrain_vl_right/configs/visual_bert/masked_coco/fifty_pc.yaml)
- [Masked COCO 10%](projects/pretrain_vl_right/configs/visual_bert/masked_coco/ten_pc.yaml)

#### VisualBERT Masked VQA2

- [Masked VQA2 100%](projects/visual_bert/configs/masked_vqa2/pretrain.yaml)
- [Masked VQA2 50%](projects/pretrain_vl_right/configs/visual_bert/masked_vqa2/fifty_pc.yaml)
- [Masked VQA2 10%](projects/pretrain_vl_right/configs/visual_bert/masked_vqa2/ten_pc.yaml)

#### VisualBERT Masked Conceptual Captions

- [Masked CC 100%](projects/visual_bert/configs/masked_conceptual_captions/pretrain.yaml)
- [Masked CC 50%](projects/pretrain_vl_right/configs/visual_bert/masked_conceptual_captions/half.yaml)
- [Masked CC 10% (CC Small 100%)](projects/pretrain_vl_right/configs/visual_bert/masked_conceptual_captions/small.yaml)
- [Masked CC Small 50% (CC Small 50%)](projects/pretrain_vl_right/configs/visual_bert/masked_conceptual_captions/small_fifty_pc.yaml)
- [Masked CC Small 10% (CC Small 10%)](projects/pretrain_vl_right/configs/visual_bert/masked_conceptual_captions/small_fifty_pc.yaml)
- [Masked CC Generated 100%](projects/pretrain_vl_right/configs/visual_bert/masked_conceptual_captions/full_coco_generated.yaml)
- [Masked CC Generated 50%](projects/pretrain_vl_right/configs/visual_bert/masked_conceptual_captions/half_coco_generated.yaml)
- [Masked CC Generated 10%](projects/pretrain_vl_right/configs/visual_bert/masked_conceptual_captions/small_coco_generated.yaml)

#### ViLBERT Masked COCO

- [Masked COCO 100%](projects/vilbert/configs/masked_coco/pretrain.yaml)
- [Masked COCO 50%](projects/pretrain_vl_right/configs/vilbert/masked_coco/fifty_pc.yaml)
- [Masked COCO 10%](projects/pretrain_vl_right/configs/vilbert/masked_coco/ten_pc.yaml)

#### ViLBERT Masked VQA2

- [Masked VQA2 100%](projects/vilbert/configs/masked_vqa2/pretrain.yaml)
- [Masked VQA2 50%](projects/pretrain_vl_right/configs/vilbert/masked_vqa2/fifty_pc.yaml)
- [Masked VQA2 10%](projects/pretrain_vl_right/configs/vilbert/masked_vqa2/ten_pc.yaml)

#### ViLBERT Masked Conceptual Captions

- [Masked CC 100%](projects/vilbert/configs/masked_conceptual_captions/pretrain.yaml)
- [Masked CC 50%](projects/pretrain_vl_right/configs/vilbert/masked_conceptual_captions/half.yaml)
- [Masked CC 10% (CC Small 100%)](projects/pretrain_vl_right/configs/vilbert/masked_conceptual_captions/small.yaml)
- [Masked CC Small 50% (CC Small 50%)](projects/pretrain_vl_right/configs/vilbert/masked_conceptual_captions/small_fifty_pc.yaml)
- [Masked CC Small 10% (CC Small 10%)](projects/pretrain_vl_right/configs/vilbert/masked_conceptual_captions/small_fifty_pc.yaml)
- [Masked CC Generated 100%](projects/pretrain_vl_right/configs/vilbert/masked_conceptual_captions/full_coco_generated.yaml)
- [Masked CC Generated 50%](projects/pretrain_vl_right/configs/vilbert/masked_conceptual_captions/half_coco_generated.yaml)
- [Masked CC Generated 10%](projects/pretrain_vl_right/configs/vilbert/masked_conceptual_captions/small_coco_generated.yaml)
