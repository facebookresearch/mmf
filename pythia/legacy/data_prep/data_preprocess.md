#### VQA v2.0
 
Download dataset 
```bash
cd ../
mkdir -p orig_data/vqa_v2.0
cd orig_data/vqa_v2.0
./../../data_prep/vqa_v2.0/download_vqa_2.0.sh

```

Preprocess dataset
```bash
cd ../../VQA_suite 
mkdir data

export PYTHONPATH=.

python data_prep/vqa_v2.0/extract_vocabulary.py \
--input_files ../orig_data/vqa_v2.0/v2_OpenEnded_mscoco_train2014_questions.json \
 ../orig_data/vqa_v2.0/v2_OpenEnded_mscoco_val2014_questions.json \
 ../orig_data/vqa_v2.0/v2_OpenEnded_mscoco_test2015_questions.json \
--out_dir data/

python data_prep/vqa_v2.0/process_answers.py \
--annotation_file ../orig_data/vqa_v2.0/v2_mscoco_train2014_annotations.json \
--val_annotation_file ../orig_data/vqa_v2.0/v2_mscoco_val2014_annotations.json  \
--out_dir data/ --min_freq 9

python data_prep/vqa_v2.0/extract_word_glove_embedding.py  \
--vocabulary_file data/vocabulary_vqa.txt  \
--glove_file ../orig_data/vqa_v2.0/glove/glove.6B.300d.txt \
--out_dir data/

python data_prep/vqa_v2.0/build_vqa_2.0_imdb.py --data_dir ../orig_data/vqa_v2.0/ --out_dir data/

```

Download image features
```bash
cd data/
wget https://dl.fbaipublicfiles.com/pythia/features/rcnn_10_100.tar.gz
wget https://dl.fbaipublicfiles.com/pythia/features/detectron.tar.gz
gunzip rcnn_10_100.tar.gz 
tar -xvf rcnn_10_100.tar
rm -f rcnn_10_100.tar

gunzip detectron.tar.gz
tar -xvf detectron.tar
rm -f detectron.tar
``` 
### Extract Image Features

We use detectron to extract image features. Setting up [detectron](https://github.com/facebookresearch/Detectron) 
and copy [tools/extract_features.py](tools/extract_features.py) to detectron for extracting features


Feature extraction works best with commit #3a38b7b of [detectron](https://github.com/facebookresearch/Detectron)
and #0dd3284 of [caffe2](https://github.com/caffe2/caffe2)


download the pretrained detectron model
```bash
wget https://dl.fbaipublicfiles.com/pythia/detectron_model/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl
wget https://dl.fbaipublicfiles.com/pythia/detectron_model/e2e_faster_rcnn_X-101-64x4d-FPN_1x_MLP_2048_FPN_512.yaml

$INPUT_DIR = /path/to/your/input/image or directory

python extract_features.py --cfg e2e_faster_rcnn_X-101-64x4d-FPN_1x_MLP_2048_FPN_512.yaml \
--wts FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl \
--min_bboxes 100 --max_bboxes 100 \
--feat_name gpu_0/fc6 \
--output_dir ~/temp_out $INPUT_DIR
```
