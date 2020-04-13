# Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA

This repository contains the code for M4C model from the following paper, released under the MMF:

* R. Hu, A. Singh, T. Darrell, M. Rohrbach, *Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA*. in CVPR, 2020 ([PDF](https://arxiv.org/pdf/1911.06258.pdf))
```
@inproceedings{hu2020iterative,
  title={Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA},
  author={Hu, Ronghang and Singh, Amanpreet and Darrell, Trevor and Rohrbach, Marcus},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

Project Page: http://ronghanghu.com/m4c

## Installation

Clone this repository, and build it with the following command.
```
cd ~/mmf
python setup.py build develop
```
This will install all M4C dependencies such as `pytorch-transformers` and `editdistance` and will also compile the python interface for PHOC features.

## Getting Data

This repo supports training and evaluation of the M4C model under three datasets: TextVQA, ST-VQA, and OCR-VQA. Please follow the [MMF documentation](https://learnpythia.readthedocs.io/en/latest/tutorials/quickstart.html#getting-data) to get data for each dataset.

Below are the download links to the vocabulary files, imdbs and features of each dataset. **Note that imdbs should be extracted under `data/imdb/`. All other files should be extracted under `data/`.** For the ST-VQA dataset, we notice that many images from COCO-Text in the downloaded ST-VQA data (around 1/3 of all images) are resized to 256×256 for unknown reasons, which degrades the image quality and distorts their aspect ratios. In the released object and OCR features below, we replaced these images with their original versions from COCO-Text as inputs to object detection and OCR systems.

The released imdbs contain OCR results and normalized bounding boxes (i.e. in the range of `[0,1]`) of each detected objects (under `obj_normalized_boxes` key) and OCR tokens (under `ocr_normalized_boxes` key). Note that the answers in ST-VQA and OCR-VQA imdbs are tiled (duplicated) to 10 answers per question to make its format consistent with the TextVQA imdbs.

For the TextVQA dataset, the downloaded file contains both imdbs with the Rosetta-en OCRs (better performance) and imdbs with Rosetta-ml OCRs (same OCR results as in the previous [LoRRA](http://openaccess.thecvf.com/content_CVPR_2019/papers/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.pdf) model). Please download the corresponding OCR feature files.

| Datasets      | M4C Vocabs | M4C ImDBs | Object Faster R-CNN Features | OCR Faster R-CNN Features |
|--------------|-----|-----|-------------------------------|-------------------------------|
| TextVQA      | [All Vocabs](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_vocabs.tar.gz) | [TextVQA ImDB](https://dl.fbaipublicfiles.com/pythia/m4c/data/imdb/m4c_textvqa.tar.gz) | [Open Images](https://dl.fbaipublicfiles.com/pythia/features/open_images.tar.gz) | [TextVQA Rosetta-en OCRs](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_textvqa_ocr_en_frcn_features.tar.gz), [TextVQA Rosetta-ml OCRs](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_textvqa_ocr_ml_frcn_features.tar.gz) |
| ST-VQA      | [All Vocabs](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_vocabs.tar.gz) | [ST-VQA ImDB](https://dl.fbaipublicfiles.com/pythia/m4c/data/imdb/m4c_stvqa.tar.gz) | [ST-VQA Objects](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_stvqa_obj_frcn_features.tar.gz) | [ST-VQA Rosetta-en OCRs](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_stvqa_ocr_en_frcn_features.tar.gz) |
| OCR-VQA      | [All Vocabs](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_vocabs.tar.gz) | [OCR-VQA ImDB](https://dl.fbaipublicfiles.com/pythia/m4c/data/imdb/m4c_ocrvqa.tar.gz) | [OCR-VQA Objects](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_ocrvqa_obj_frcn_features.tar.gz) | [OCR-VQA Rosetta-en OCRs](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_ocrvqa_ocr_en_frcn_features.tar.gz) |

In addition, you also need to download the detectron weights to use Faster R-CNN features:
```
# Download detectron weights
cd data/
wget http://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf detectron_weights.tar.gz
cd ..
```

Note that the object Faster R-CNN features are extracted with [`extract_features_vmb.py`](../../pythia/scripts/features/extract_features_vmb.py) and the OCR Faster R-CNN features are extracted with [`extract_ocr_frcn_feature.py`](../../projects/M4C/scripts/extract_ocr_frcn_feature.py).

## Pretrained M4C Models

We release the following pretrained models for M4C on three datasets: TextVQA, ST-VQA, and OCR-VQA.

For the TextVQA dataset, we release three versions: M4C trained with ST-VQA as additional data (our best model) with Rosetta-en, M4C trained on TextVQA alone with Rosetta-en, and M4C trained on TextVQA alone with Rosetta-ml (same OCR results as in the previous [LoRRA](http://openaccess.thecvf.com/content_CVPR_2019/papers/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.pdf) model).

| Datasets  | Config Files (under `configs/vqa/`)         | Pretrained Models | Metrics                     | Notes                         |
|--------|------------------|----------------------------|-------------------------------|-------------------------------|
| TextVQA (`m4c_textvqa`) | `m4c_textvqa/m4c_with_stvqa.yaml` | [`m4c_textvqa_m4c_with_stvqa`](https://dl.fbaipublicfiles.com/pythia/m4c/m4c_release_models/m4c_textvqa/m4c_textvqa_m4c_with_stvqa.ckpt) | val accuracy - 40.55%; test accuracy - 40.46% | Rosetta-en OCRs; ST-VQA as additional data |
| TextVQA (`m4c_textvqa`) | `m4c_textvqa/m4c.yaml` | [`m4c_textvqa_m4c`](https://dl.fbaipublicfiles.com/pythia/m4c/m4c_release_models/m4c_textvqa/m4c_textvqa_m4c.ckpt) | val accuracy - 39.40%; test accuracy - 39.01% | Rosetta-en OCRs |
| TextVQA (`m4c_textvqa`) | `m4c_textvqa/m4c_ocr_ml.yaml` | [`m4c_textvqa_m4c_ocr_ml`](https://dl.fbaipublicfiles.com/pythia/m4c/m4c_release_models/m4c_textvqa/m4c_textvqa_m4c_ocr_ml.ckpt) | val accuracy - 37.06% | Rosetta-ml OCRs |
| ST-VQA (`m4c_stvqa`)  | `m4c_stvqa/m4c.yaml` | [`m4c_stvqa_m4c`](https://dl.fbaipublicfiles.com/pythia/m4c/m4c_release_models/m4c_stvqa/m4c_stvqa_m4c.ckpt) | val ANLS - 0.472 (accuracy - 38.05%); test ANLS - 0.462 | Rosetta-en OCRs |
| OCR-VQA (`m4c_ocrvqa`) | `m4c_ocrvqa/m4c.yaml` | [`m4c_ocrvqa_m4c`](https://dl.fbaipublicfiles.com/pythia/m4c/m4c_release_models/m4c_ocrvqa/m4c_ocrvqa_m4c.ckpt) | val accuracy - 63.52%; test accuracy - 63.87% | Rosetta-en OCRs |

## Training and Evaluation

Please follow the [Pythia documentation](https://learnpythia.readthedocs.io/en/latest/tutorials/quickstart.html#training) for the training and evaluation of the M4C model on each dataset.

For example:

1) to train the M4C model on the TextVQA training set:
```
# Distributed Data Parallel (on a 4-GPU machine)
# (change `--nproc_per_node 4` to the actual GPU number on your machine)
python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c \
--config configs/vqa/m4c_textvqa/m4c.yaml \
--save_dir save/m4c \
training.distributed True

# alternative: Data Parallel (slower, but results should be the same)
python tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c \
--config configs/vqa/m4c_textvqa/m4c.yaml \
--save_dir save/m4c \
training.data_parallel True
```
(Replace `m4c_textvqa` with other datasets and `configs/vqa/m4c_textvqa/m4c.yaml` with other config files to train with other datasets and configurations. See the table above. You can also specify a different path to `--save_dir` to save to a location you prefer.)

2) to evaluate the pretrained M4C model locally on the TextVQA validation set (assuming the pretrained model is downloaded to `data/models/m4c_textvqa_m4c.ckpt`):
```
python tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c \
--config configs/vqa/m4c_textvqa/m4c.yaml \
--save_dir save/m4c \
--run_type val \
--resume_file data/models/m4c_textvqa_m4c.ckpt
```
(Note: use `--resume 1` instead of `--resume_file data/models/m4c_textvqa_m4c.ckpt` to evaluated your trained snapshots.)

3) to generate the EvalAI prediction files for the TextVQA test set (assuming the pretrained model is downloaded to `data/models/m4c_textvqa_m4c.ckpt`):
```
python tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c \
--config configs/vqa/m4c_textvqa/m4c.yaml \
--save_dir save/m4c \
--run_type inference --evalai_inference 1 \
--resume_file data/models/m4c_textvqa_m4c.ckpt
```
(Note: use `--resume 1` instead of `--resume_file data/models/m4c_textvqa_m4c.ckpt` to evaluated your trained snapshots. For running inference on val set, use `--run_type val` and rest of the arguments remain same.)
