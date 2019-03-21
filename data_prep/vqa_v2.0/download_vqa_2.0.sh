#!/usr/bin/env bash


# GloVe Vectors
wget  http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove
rm glove.6B.zip


##VQA2.0

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip

unzip v2_Annotations_Train_mscoco.zip
rm v2_Annotations_Train_mscoco.zip

unzip v2_Annotations_Val_mscoco.zip
rm v2_Annotations_Val_mscoco.zip

unzip v2_Questions_Train_mscoco.zip
rm v2_Questions_Train_mscoco.zip

unzip v2_Questions_Val_mscoco.zip
rm v2_Questions_Val_mscoco.zip

unzip v2_Questions_Test_mscoco.zip
rm v2_Questions_Test_mscoco.zip

unzip v2_Complementary_Pairs_Train_mscoco.zip
rm v2_Complementary_Pairs_Train_mscoco.zip

unzip v2_Complementary_Pairs_Val_mscoco.zip
rm v2_Complementary_Pairs_Val_mscoco.zip


### get minival and val2train
wget https://dl.fbaipublicfiles.com/pythia/data/v2_OpenEnded_mscoco_minival2014_questions.json
wget https://dl.fbaipublicfiles.com/pythia/data/v2_OpenEnded_mscoco_val2train2014_questions.json
