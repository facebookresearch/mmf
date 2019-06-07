# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import logging
import os
from config.collections import AttrDict
from config.function_config_lib import ModelParPair

logger = logging.getLogger(__name__)

__C = AttrDict()
cfg = __C


# --------------------------------------------------------------------------- #
# running model options: string, options: train, train+predict
# --------------------------------------------------------------------------- #
__C.run = "train+predict"
__C.exp_name = "baseline"


# --------------------------------------------------------------------------- #
# data options:
# --------------------------------------------------------------------------- #
__C.data = AttrDict()
__C.data.dataset = "vqa_2.0"
__C.data.num_workers = 5
__C.data.batch_size = 512
__C.data.image_depth_first = False
__C.data.question_max_len = 14
__C.data.image_fast_reader = True
__C.data.image_max_loc = 100

__C.data.data_root_dir = "data"

__C.data.vocab_question_file = "vocabulary_vqa.txt"
__C.data.vocab_answer_file = "answers_vqa.txt"

__C.data.image_feat_train = ["rcnn_10_100/vqa/train2014",
                             "rcnn_10_100/vqa/val2014"]
__C.data.imdb_file_train = ["imdb/imdb_train2014.npy",
                            "imdb/imdb_val2train2014.npy"]

__C.data.imdb_file_val = ["imdb/imdb_minival2014.npy"]
__C.data.image_feat_val = ["rcnn_10_100/vqa/val2014"]

__C.data.imdb_file_test = ["imdb/imdb_test2015.npy"]
__C.data.image_feat_test = ["rcnn_10_100/vqa/test2015"]


# --------------------------------------------------------------------------- #
# training_parameters options:
# --------------------------------------------------------------------------- #
__C.training_parameters = AttrDict()
__C.training_parameters.report_interval = 100
__C.training_parameters.snapshot_interval = 1000
__C.training_parameters.clip_norm_mode = 'all'
__C.training_parameters.max_grad_l2_norm = 0.25
__C.training_parameters.wu_factor = 0.2
__C.training_parameters.wu_iters = 1000
__C.training_parameters.max_iter = 12000
__C.training_parameters.lr_steps = [5000, 7000, 9000, 11000]
__C.training_parameters.lr_ratio = 0.1


# --------------------------------------------------------------------------- #
# loss options:
# --------------------------------------------------------------------------- #
__C.loss = 'logitBCE'


# --------------------------------------------------------------------------- #
# optimizer options:
# --------------------------------------------------------------------------- #
__C.optimizer = ModelParPair('Adamax')


# --------------------------------------------------------------------------- #
# model options: Note default is our
# --------------------------------------------------------------------------- #
__C.model = AttrDict()
__C.model.image_feat_dim = 2048
__C.model.question_embedding = [ModelParPair("att_que_embed")]
__C.model.image_feature_encoding = [ModelParPair('default_image')]
__C.model.image_embedding_models = []
__C.model.modal_combine = ModelParPair('non_linear_elmt_multiply')
__C.model.classifier = ModelParPair('logit_classifier')

top_down_bottom_up = AttrDict()
top_down_bottom_up.modal_combine = ModelParPair('non_linear_elmt_multiply')
top_down_bottom_up.transform = ModelParPair('linear_transform')
top_down_bottom_up.normalization = 'softmax'

__C.model.image_embedding_models.append(top_down_bottom_up)

# --------------------------------------------------------------------------- #
# failure prediction options:
# --------------------------------------------------------------------------- #
failure_predictor = AttrDict()
failure_predictor.hidden_1 = 0
failure_predictor.hidden_2 = 512
failure_predictor.answer_hidden_size = 256
failure_predictor.dropout = 0.5
failure_predictor.feat_combine = 'iq'

__C.model.failure_predictor = failure_predictor

# --------------------------------------------------------------------------- #
# question generator options:
# --------------------------------------------------------------------------- #
question_consistency = AttrDict()
question_consistency.hidden_1 = 0
question_consistency.attended = False
question_consistency.cycle = False
question_consistency.vqa_gating = False        # don't use VQA gating by default
question_consistency.activation_iter = 10e10   # Never activated by def
question_consistency.gating_th = 0             # Pass all questions by default
question_consistency.hidden_size = 512         # hidden size of LSTM
question_consistency.embed_size = 300          # embedding size of image, answer feats
question_consistency.ans_embed_hidden_size = 1000  # hidden state of answer embedding layer
question_consistency.image_feature_in_size = 2048  # input image feat size


__C.model.question_consistency = question_consistency

# --------------------------------------------------------------------------- #
# cycle-consistency options:
# --------------------------------------------------------------------------- #
__C.training_parameters.fp_lr = 0.001    # Default lr for failure predictor
__C.training_parameters.qc_lr = 0.001    # Default lr for question generation
__C.training_parameters.fp_lambda = 1.0  # lr multiplier  for failure predictor loss
__C.training_parameters.qc_lambda = 1.0  # lr_multiplier for question generation loss
__C.training_parameters.cc_lambda = 0.5  # lr_multiplier for answer consistency loss

