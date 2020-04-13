# TextCaps: a Dataset for Image Captioning with Reading Comprehension

This repository contains the code for M4C-Captioner model, released under the MMF.

Project Page: https://textvqa.org/textcaps/

## Installation

Clone this repository, and build it with the following command.
```
cd ~/mmf
python setup.py build develop
# install pycocoevalcap
# use the repo below instead of https://github.com/tylin/coco-caption
# note: you also need to have java on your machine
pip install git+https://github.com/ronghanghu/coco-caption.git@python23
```
This will install all M4C-Captioner dependencies such as `pytorch-transformers`, `editdistance` and `pycocoevalcap`, and will also compile the python interface for PHOC features.

**Note that java is required for [`pycocoevalcap`](https://github.com/ronghanghu/coco-caption)**.

## Getting Data

This repo supports training and evaluation of the M4C-Captioner model. Please follow the [MMF documentation](https://learnpythia.readthedocs.io/en/latest/tutorials/quickstart.html#getting-data) to get data for each dataset.

Below are the download links to the vocabulary files, imdbs and features. **Note that imdbs should be extracted under `data/imdb/`. All other files should be extracted under `data/`.**

The released imdbs contain OCR results and normalized bounding boxes (i.e. in the range of `[0,1]`) of each detected objects (under `obj_normalized_boxes` key) and OCR tokens (under `ocr_normalized_boxes` key). These OCR tokens are extracted with Rosetta-en.

| Datasets      | M4C-Captioner Vocabs | M4C-Captioner ImDBs | Object Faster R-CNN Features | OCR Faster R-CNN Features |
|--------------|-----|-----|-------------------------------|-------------------------------|
| TextCaps      | [All Vocabs](https://dl.fbaipublicfiles.com/pythia/m4c_captioner/data/m4c_captioner_vocabs.tar.gz) | [TextCaps ImDB](https://dl.fbaipublicfiles.com/pythia/m4c_captioner/data/imdb/m4c_textcaps.tar.gz) | [Open Images](https://dl.fbaipublicfiles.com/pythia/features/open_images.tar.gz) | [TextCaps Rosetta-en OCRs](https://dl.fbaipublicfiles.com/pythia/m4c/data/m4c_textvqa_ocr_en_frcn_features.tar.gz) |
| COCO      | [All Vocabs](https://dl.fbaipublicfiles.com/pythia/m4c_captioner/data/m4c_captioner_vocabs.tar.gz) | [COCO ImDB](https://dl.fbaipublicfiles.com/pythia/m4c_captioner/data/imdb/m4c_coco.tar.gz) | [COCO Objects](https://dl.fbaipublicfiles.com/pythia/m4c_captioner/data/coco_obj_frcn_features.tar.gz) | [COCO Rosetta-en OCRs](https://dl.fbaipublicfiles.com/pythia/m4c_captioner/data/coco_ocr_en_frcn_features.tar.gz) |

In addition, you also need to download the detectron weights to use Faster R-CNN features:
```
# Download detectron weights
cd data/
wget http://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf detectron_weights.tar.gz
cd ..
```

Note that the object Faster R-CNN features are extracted with [`extract_features_vmb.py`](../../pythia/scripts/features/extract_features_vmb.py) and the OCR Faster R-CNN features are extracted with [`extract_ocr_frcn_feature.py`](../../projects/M4C/scripts/extract_ocr_frcn_feature.py).

## Pretrained M4C-Captioner Models

We release the following pretrained model for M4C-Captioner for the TextCaps dataset:

| Datasets  | Config Files (under `configs/captioning/`)         | Pretrained Models | Metrics                     |
|--------|------------------|----------------------------|-------------------------------|
| TextCaps (`m4c_textcaps`) | `m4c_textcaps/m4c_captioner.yaml` | [`download`](https://dl.fbaipublicfiles.com/pythia/m4c_captioner/m4c_captioner_release_models/m4c_textcaps/m4c_textcaps_m4c_captioner.ckpt) | val CIDEr -- 89.6; test CIDEr -- 81.0 |

## Training and Evaluating M4C-Captioner

Please follow the [MMF documentation](https://learnpythia.readthedocs.io/en/latest/tutorials/quickstart.html#training) for the training and evaluation of the M4C-Captioner model on each dataset.

For example:

1) to train the M4C-Captioner model on the TextCaps training set:
```
# Distributed Data Parallel (on a 4-GPU machine)
# (change `--nproc_per_node 4` to the actual GPU number on your machine)
python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/m4c_captioner.yaml \
--save_dir save/m4c_captioner \
training.distributed True

# alternative: Data Parallel (slower, but results should be the same)
python tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/m4c_captioner.yaml \
--save_dir save/m4c_captioner \
training.data_parallel True
```
(You can also specify a different path to `--save_dir` to save to a location you prefer. Replace `configs/captioning/m4c_textcaps/m4c_captioner.yaml` with `configs/captioning/m4c_textcaps/m4c_captioner_without_ocr.yaml` to train M4C-Captioner without using OCR inputs as an ablation study.)

2) to generate prediction json files for the TextCaps (assuming the pretrained model is downloaded to `data/models/m4c_textcaps_m4c_captioner.ckpt`):
```
# generate predictions on val
python tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/m4c_captioner.yaml \
--save_dir save/m4c_captioner \
--run_type val --evalai_inference 1 \
--resume_file data/models/m4c_textcaps_m4c_captioner.ckpt

# generate predictions on test
python tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/m4c_captioner.yaml \
--save_dir save/m4c_captioner \
--run_type inference --evalai_inference 1 \
--resume_file data/models/m4c_textcaps_m4c_captioner.ckpt
```
(Note: use `--resume 1` instead of `--resume_file data/models/m4c_textcaps_m4c_captioner.ckpt` to evaluated your trained snapshots.)

Afterwards, use `projects/M4C_Captioner/scripts/textcaps_eval.py` to evaluate the prediction json file. For example:
```
python projects/M4C_Captioner/scripts/textcaps_eval.py --set val --pred_file YOUR_VAL_PREDICTION_FILE
```
For test set evaluation, please submit to the TextCaps EvalAI server.

### Training M4C-Captioner on COCO and Evaluating on TextCaps (or COCO)

1) to train the M4C-Captioner model on the COCO training set:
```
# Distributed Data Parallel (on a 4-GPU machine)
# (change `--nproc_per_node 4` to the actual GPU number on your machine)
python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/m4c_captioner_coco.yaml \
--save_dir save/m4c_captioner_coco \
training.distributed True
```

2) to generate prediction json files for the TextCaps:
```
# generate predictions on TextCaps val
python tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/m4c_captioner_coco_eval_on_textcaps.yaml \
--run_type val --evalai_inference 1 \
--save_dir save/m4c_captioner_coco \
--resume 1

# generate predictions on TextCaps test
python tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/m4c_captioner_coco_eval_on_textcaps.yaml \
--run_type inference --evalai_inference 1 \
--save_dir save/m4c_captioner_coco \
--resume 1
```
and evaluate the prediction files with `projects/M4C_Captioner/scripts/textcaps_eval.py`. For example:
```
python projects/M4C_Captioner/scripts/textcaps_eval.py --set val --pred_file YOUR_VAL_PREDICTION_FILE
```
For test set evaluation, please submit to the TextCaps EvalAI server.

3) to generate prediction json files for the COCO Karpathy val split (in paper supplemental):
```
# generate predictions on COCO Karpathy val
python tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/m4c_captioner_coco.yaml \
--run_type val --evalai_inference 1 \
--save_dir save/m4c_captioner_coco \
--resume 1
```
and evaluate the prediction files with `projects/M4C_Captioner/scripts/coco_eval.py`. For example:
```
python projects/M4C_Captioner/scripts/coco_eval.py \
--set karpathy_val \
--pred_file YOUR_COCO_KARPATHY_VAL_PREDICTION_FILE
```

### M4C-Captioner trained on joint COCO + TextCaps (in paper supplemental)

1) to train the M4C-Captioner model on the joint TextCaps + COCO:
```
# Distributed Data Parallel (on a 4-GPU machine)
# (change `--nproc_per_node 4` to the actual GPU number on your machine)
python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/m4c_captioner_coco_textcaps_joint.yaml \
--save_dir save/m4c_captioner_coco_textcaps_joint \
training.distributed True
```

2) to generate prediction json files for the COCO Karpathy val split (in paper supplemental):
```
# generate predictions on COCO Karpathy val
python tools/run.py --tasks captioning --datasets m4c_textcaps --model m4c_captioner \
--config configs/captioning/m4c_textcaps/m4c_captioner_coco_textcaps_joint.yaml \
--run_type val --evalai_inference 1 \
--save_dir save/m4c_captioner_coco_textcaps_joint \
--resume 1
```
and evaluate the prediction files with `projects/M4C_Captioner/scripts/coco_eval.py` as above.

## Training and Evaluating BUTD on TextCaps

We load the BUTD model based on the COCO BUTD config (but replaced the imdb and vocab files with the TextCaps dataset ones).

To train the BUTD model on TextCaps:
```
# Data Parallel (Distributed Data Parallel doesn't work with BUTD for now)
python tools/run.py --tasks captioning --datasets coco --model butd \
--config configs/captioning/m4c_textcaps/butd.yaml \
--save_dir save/butd \
training.data_parallel True
```
(Note that although we use `--datasets coco` here, it actually loads TextCaps imdb files. So we are training on TextCaps, not COCO.)

To evaluate the trained BUTD model on TextCaps:
```
# generate predictions on val
python tools/run.py --tasks captioning --datasets coco --model butd \
--config configs/captioning/m4c_textcaps/butd_beam_search.yaml \
--save_dir save/butd \
--run_type val --evalai_inference 1 \
--resume 1

# generate predictions on test
python tools/run.py --tasks captioning --datasets coco --model butd \
--config configs/captioning/m4c_textcaps/butd_beam_search.yaml \
--save_dir save/butd \
--run_type inference --evalai_inference 1 \
--resume 1
```
and evaluate the prediction files with `projects/M4C_Captioner/scripts/textcaps_eval.py`. For example:
```
python projects/M4C_Captioner/scripts/textcaps_eval.py --set val --pred_file YOUR_VAL_PREDICTION_FILE
```
For test set evaluation, please submit to the TextCaps EvalAI server.

### Evaluating COCO-pretrained BUTD on TextCaps

On the TextCaps dataset, one can also directly evaluate the BUTD model pretrained on COCO captioning dataset:
```
# download COCO-pretrained BUTD model
wget https://dl.fbaipublicfiles.com/pythia/pretrained_models/coco_captions/butd.pth \
-O data/coco_pretrained_butd.pth

# generate predictions on val
python tools/run.py --tasks captioning --datasets coco --model butd \
--config configs/captioning/m4c_textcaps/butd_eval_pretrained_coco_model.yaml \
--save_dir save/butd_eval_pretrained_coco_model \
--run_type val --evalai_inference 1 \
--resume_file data/coco_pretrained_butd.pth

# generate predictions on test
python tools/run.py --tasks captioning --datasets coco --model butd \
--config configs/captioning/m4c_textcaps/butd_eval_pretrained_coco_model.yaml \
--save_dir save/butd_eval_pretrained_coco_model \
--run_type inference --evalai_inference 1 \
--resume_file data/coco_pretrained_butd.pth
```
and evaluate the prediction files with `projects/M4C_Captioner/scripts/textcaps_eval.py`. For example:
```
python projects/M4C_Captioner/scripts/textcaps_eval.py --set val --pred_file YOUR_VAL_PREDICTION_FILE
```
For test set evaluation, please submit to the TextCaps EvalAI server.
