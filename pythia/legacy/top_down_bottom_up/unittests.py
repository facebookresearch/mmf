# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import unittest

import numpy as np
import torch
from torch.autograd import Variable

from global_variables.global_variables import use_cuda
from top_down_bottom_up.classifier import logit_classifier
from top_down_bottom_up.image_embedding import image_embedding
from top_down_bottom_up.question_embeding import QuestionEmbeding
from top_down_bottom_up.top_down_bottom_up_model import \
    top_down_bottom_up_model


class Test_top_down_bottom_up_model(unittest.TestCase):
    def test_classifier(self):
        batch_size = 12
        joint_embedding_dim = 10
        num_ans_candidates = 20
        text_embeding_dim = 64
        image_embedding_dim = 32

        my_classifier = logit_classifier(
            joint_embedding_dim,
            num_ans_candidates,
            image_embedding_dim,
            text_embeding_dim,
        )
        joint_embedding = Variable(torch.randn(batch_size, joint_embedding_dim))
        res = my_classifier(joint_embedding)
        self.assertEqual((12, 20), res.shape)

    def test_classifier_batch_size_1(self):
        batch_size = 1
        joint_embedding_dim = 10
        num_ans_candidates = 20
        text_embeding_dim = 64
        image_embedding_dim = 32

        my_classifier = logit_classifier(
            joint_embedding_dim,
            num_ans_candidates,
            image_embedding_dim,
            text_embeding_dim,
        )
        joint_embedding = Variable(torch.randn(batch_size, joint_embedding_dim))
        res = my_classifier(joint_embedding)
        self.assertEqual((1, 20), res.shape)

    def test_question_embedding(self):
        num_vocab = 20
        embedding_dim = 300
        lstm_dim = 512
        lstm_layer = 1
        dropout = 0.1
        batch_first = True
        batch_size = 32
        question_len = 10
        my_word_embedding_model = QuestionEmbeding(
            num_vocab, embedding_dim, lstm_dim, lstm_layer, dropout, batch_first
        )
        my_word_embedding_model = (
            my_word_embedding_model.cuda() if use_cuda else my_word_embedding_model
        )
        input_txt = Variable(
            torch.rand(batch_size, question_len).type(torch.LongTensor) % num_vocab
        )
        input_txt = input_txt.cuda() if use_cuda else input_txt
        embedding = my_word_embedding_model(input_txt, batch_first=True)
        self.assertEqual((32, 512), embedding.shape)

    def test_image_embedding(self):
        image_feat_dim = 40
        txt_embedding_dim = 50
        hidden_size = 30
        num_of_loc = 5
        batch_size = 16
        my_image_embeding = image_embedding(
            image_feat_dim, txt_embedding_dim, hidden_size
        )
        image_feat = Variable(torch.randn(batch_size, num_of_loc, image_feat_dim))
        txt = Variable(torch.randn(batch_size, txt_embedding_dim))
        res = my_image_embeding(image_feat, txt)
        self.assertEqual((batch_size, image_feat_dim), res.shape)

    def test_model(self):
        image_feat_dim = 40
        txt_embedding_dim = 300
        lstm_dim = 512
        hidden_size = 30
        num_of_loc = 5
        batch_size = 16
        num_vocab = 60
        num_ans_candidates = 35
        joint_embedding_dim = 500
        question_len = 13
        batch_first = True
        image_embedding_model = image_embedding(image_feat_dim, lstm_dim, hidden_size)
        question_embedding_model = QuestionEmbeding(
            num_vocab,
            txt_embedding_dim,
            lstm_dim,
            lstm_layer=2,
            dropout=0.1,
            batch_first=batch_first,
        )
        my_classifier = logit_classifier(
            joint_embedding_dim, num_ans_candidates, image_feat_dim, txt_embedding_dim
        )
        loss = torch.nn.CrossEntropyLoss()

        my_model = top_down_bottom_up_model(
            image_embedding_model, question_embedding_model, my_classifier, loss
        )
        image_feat = np.random.rand(batch_size, num_of_loc, image_feat_dim)
        input_txt = Variable(
            torch.rand(batch_size, question_len).type(torch.LongTensor) % num_vocab
        )
        res = my_model(image_feat, input_txt, batch_first)
        self.assertEqual((batch_size, num_ans_candidates), res.shape)


if __name__ == "__main__":
    unittest.main()
