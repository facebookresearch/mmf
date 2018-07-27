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
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/rcnn_10_100.tar.gz
wget https://s3-us-west-1.amazonaws.com/pythia-vqa/data/detectron.tar.gz
gunzip rcnn_10_100.tar.gz 
tar -xvf rcnn_10_100.tar
rm -f rcnn_10_100.tar

gunzip detectron.tar.gz
tar -xvf detectron.tar
rm -f detectron.tar


``` 