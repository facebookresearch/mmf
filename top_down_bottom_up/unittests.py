# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import unittest
import torch
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))  # add parent for imports

from config.collections import AttrDict
from config.function_config_lib import ModelParPair
from top_down_bottom_up.image_attention import build_image_attention_module
from top_down_bottom_up.multi_modal_combine import build_modal_combine_module
from top_down_bottom_up.post_combine_transform import \
    build_post_combine_transform
from torch.autograd import Variable
from top_down_bottom_up.question_embeding import QuestionEmbeding, \
    AttQuestionEmbedding
from top_down_bottom_up.image_embedding import image_embedding
from top_down_bottom_up.image_feature_encoding import \
    build_image_feature_encoding
from top_down_bottom_up.classifier import logit_classifier
from top_down_bottom_up.top_down_bottom_up_model \
    import top_down_bottom_up_model, vqa_multi_modal_model
import numpy as np
import pickle
from config.config import cfg
from global_variables.global_variables import use_cuda


class TestVqaModel(unittest.TestCase):

    def test_classifier(self):
        batch_size = 12
        joint_embedding_dim = 10
        num_ans_candidates = 20
        text_embeding_dim = 64
        image_embedding_dim = 32

        my_classifier = logit_classifier(joint_embedding_dim,
                                         num_ans_candidates,
                                         img_hidden_dim=image_embedding_dim,
                                         txt_hidden_dim=text_embeding_dim)
        joint_embedding = Variable(torch.randn(batch_size,
                                               joint_embedding_dim))
        res = my_classifier(joint_embedding)
        self.assertEqual((12, 20), res.shape)

    def test_classifier_batch_size_1(self):
        batch_size = 1
        joint_embedding_dim = 10
        num_ans_candidates = 20
        text_embeding_dim = 64
        image_embedding_dim = 32

        my_classifier = logit_classifier(joint_embedding_dim,
                                         num_ans_candidates,
                                         img_hidden_dim=image_embedding_dim,
                                         txt_hidden_dim=text_embeding_dim)
        joint_embedding = Variable(torch.randn(batch_size,
                                               joint_embedding_dim))
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
        my_word_embedding_model = QuestionEmbeding(num_vocab=num_vocab,
                                                   embedding_dim=embedding_dim,
                                                   LSTM_hidden_size=lstm_dim,
                                                   lstm_layer=lstm_layer,
                                                   lstm_dropout=dropout,
                                                   batch_first=batch_first)
        input_txt = Variable(torch.rand(batch_size,
                                        question_len).type(
            torch.LongTensor) % num_vocab)
        embedding = my_word_embedding_model(input_txt)
        self.assertEqual((32, 512), embedding.shape)

    def test_question_atten_embedding(self):
        num_vocab = 20
        embedding_dim = 300
        lstm_dim = 64
        lstm_layer = 1
        dropout = 0.1
        batch_first = True
        batch_size = 4
        question_len = 10
        conv1_out = 512
        conv2_out = 2
        kernel_size = 1
        padding = 0
        word_embedding_model = AttQuestionEmbedding(num_vocab=num_vocab,
                                                    embedding_dim=
                                                    embedding_dim,
                                                    LSTM_hidden_size=
                                                    lstm_dim,
                                                    LSTM_layer=lstm_layer,
                                                    dropout=dropout,
                                                    batch_first=batch_first,
                                                    conv1_out=conv1_out,
                                                    conv2_out=conv2_out,
                                                    padding=padding,
                                                    kernel_size=kernel_size)

        input_txt = Variable(torch.rand(batch_size,
                                        question_len).type(
            torch.LongTensor) % num_vocab)
        embedding = word_embedding_model(input_txt)
        self.assertEqual((4, 64 * 2), embedding.shape)

    def test_image_feature_encoding(self):
        in_dim = 128
        out_dim = 256
        weights = np.random.rand(out_dim, in_dim) * 10
        bias = np.random.rand(out_dim)
        weights_file = 'test_weights.pkl'
        bias_file = 'test_biases.pkl'
        method = "finetune_faster_rcnn_fpn_fc7"
        input = Variable(torch.randn(in_dim))
        par = AttrDict()
        par.weights_file = weights_file
        par.bias_file = bias_file
        with open(os.path.join(cfg.data.data_root_dir, weights_file), 'wb') \
                as file:
            pickle.dump(weights, file)

        with open(os.path.join(cfg.data.data_root_dir, bias_file), 'wb') \
                as file:
            pickle.dump(bias, file)
        img_encoding_model = build_image_feature_encoding(method,
                                                          par=par,
                                                          in_dim=in_dim)
        res = img_encoding_model(input)
        os.remove(os.path.join(cfg.data.data_root_dir, weights_file))
        os.remove(os.path.join(cfg.data.data_root_dir, bias_file))

        self.assertEqual((256,), res.shape)

    def test_multi_modal_combine(self):
        method = 'non_linear_elmt_multiply'
        par_pair = ModelParPair(method)
        par_pair.par['hidden_size'] = 200
        n_batch = 10
        num_img_feats = 100
        img_feat_dim = 128
        ques_emb_dim = 64
        img_feats = Variable(torch.rand(n_batch, num_img_feats, img_feat_dim))
        ques_embeds = Variable(torch.rand(n_batch, ques_emb_dim))

        modal_combine_model = build_modal_combine_module(method, par_pair.par,
                                                         img_feat_dim,
                                                         ques_emb_dim)
        res = modal_combine_model(img_feats, ques_embeds)
        self.assertEqual((n_batch, num_img_feats, 200), res.shape)

    def test_post_combine_transform(self):
        method = 'linear_transform'
        par_pair = ModelParPair(method)
        n_batch = 10
        num_img_feats = 100
        in_dim = 64
        joint_feats = Variable(torch.rand(n_batch, num_img_feats, in_dim))
        post_combine_transform = build_post_combine_transform(method,
                                                              par_pair.par,
                                                              in_dim)
        res = post_combine_transform(joint_feats)
        self.assertEqual((n_batch, num_img_feats, par_pair.par.out_dim),
                         res.shape)

    def test_image_attention_module(self):
        multi_modal_method = 'non_linear_elmt_multiply'
        post_combine_method = 'linear_transform'
        normalization_method = 'softmax'
        n_batch = 10
        num_img_feats = 100
        image_feat_dim = 128
        ques_embed_dim = 64
        attr = AttrDict()
        attr.modal_combine = ModelParPair(multi_modal_method)
        attr.transform = ModelParPair(post_combine_method)
        attr.normalization = normalization_method
        image_feats = Variable(torch.randn(n_batch,
                                           num_img_feats,
                                           image_feat_dim))

        ques_embeds = Variable(torch.randn(n_batch, ques_embed_dim))

        img_atten_model = build_image_attention_module(attr,
                                                       image_feat_dim,
                                                       ques_embed_dim)
        res = img_atten_model(image_feats, ques_embeds)
        self.assertEqual((n_batch, num_img_feats, attr.transform.par.out_dim),
                         res.shape)

    def test_image_embedding(self):
        multi_modal_method = 'non_linear_elmt_multiply'
        post_combine_method = 'linear_transform'
        normalization_method = 'softmax'
        n_batch = 10
        num_img_feats = 100
        image_feat_dim = 128
        ques_embed_dim = 64
        attr = AttrDict()
        attr.modal_combine = ModelParPair(multi_modal_method)
        attr.transform = ModelParPair(post_combine_method)
        attr.normalization = normalization_method
        image_feats = Variable(torch.randn(n_batch,
                                           num_img_feats,
                                           image_feat_dim))

        ques_embeds = Variable(torch.randn(n_batch, ques_embed_dim))

        img_atten_model = build_image_attention_module(attr,
                                                       image_feat_dim,
                                                       ques_embed_dim)
        img_embedding_model = image_embedding(img_atten_model)
        res = img_embedding_model(image_feats, ques_embeds, None)
        self.assertEqual((n_batch, image_feat_dim), res.shape)

    def test_vqa_multi_modal_model(self):
        num_vocab = 64
        num_ans_candidates = 128
        embedding_dim = 300
        lstm_dim = 512
        lstm_layer = 1
        dropout = 0.1
        batch_first = True
        n_batch = 32
        question_len = 10
        conv1_out = 512
        conv2_out = 2
        kernel_size = 1
        padding = 0
        out_dim_attn_que_embed = lstm_dim * conv2_out
        # ----------------------------------------------------------------------
        # input shape:  [n_batch, question_len]
        # output shape:  [n_batch, lstm_dim*conv2_out]
        # ----------------------------------------------------------------------
        ques_embedding_model = AttQuestionEmbedding(num_vocab=num_vocab,
                                                    embedding_dim=
                                                    embedding_dim,
                                                    LSTM_hidden_size=
                                                    lstm_dim,
                                                    LSTM_layer=lstm_layer,
                                                    dropout=dropout,
                                                    batch_first=batch_first,
                                                    conv1_out=conv1_out,
                                                    conv2_out=conv2_out,
                                                    padding=padding,
                                                    kernel_size=kernel_size)
        ques_embedding_model_list = [ques_embedding_model]

        in_dim = 128
        encoding_method = "default_image"
        par = None
        num_img_feats = 100
        out_dim_img_enc = in_dim
        # ----------------------------------------------------------------------
        # input shape:  [n_batch, num_image_feats, in_dim]
        # output shape:  [n_batch, num_image_feats, out_dim_img_enc]
        # ----------------------------------------------------------------------
        img_encoding_model = build_image_feature_encoding(encoding_method,
                                                          par=par,
                                                          in_dim=in_dim)
        img_encoding_model_list = [img_encoding_model]

        multi_modal_method = 'non_linear_elmt_multiply'
        post_combine_method = 'linear_transform'
        normalization_method = 'softmax'
        image_feat_dim = out_dim_img_enc
        ques_embed_dim = out_dim_attn_que_embed
        attr = AttrDict()
        attr.modal_combine = ModelParPair(multi_modal_method)
        attr.transform = ModelParPair(post_combine_method)
        attr.normalization = normalization_method
        out_dim_attn_img_embed = image_feat_dim * attr.transform.par.out_dim
        # ----------------------------------------------------------------------
        # inputs:  image_feats and ques_embeds
        # image_feats shape:  [n_batch, num_image_feats, image_feat_dim]
        # ques_embeds shape:  [n_batch, ques_embed_dim]
        # output shape:  [n_batch, num_image_feats, out_dim_attn_img_embed]
        # ----------------------------------------------------------------------
        img_atten_model = build_image_attention_module(attr,
                                                       image_feat_dim,
                                                       ques_embed_dim)
        img_embedding_model_list = [[image_embedding(img_atten_model)]]

        img_embed_dim = out_dim_attn_img_embed
        ques_embed_dim = out_dim_attn_que_embed
        final_modal_method = 'non_linear_elmt_multiply'
        par_pair = ModelParPair(final_modal_method)
        out_dim_modal_combine = par_pair.par.hidden_size
        # ----------------------------------------------------------------------
        # inputs:  image_embeds and ques_embeds
        # image_embeds shape: [n_batch, img_embed_dim]
        # ques_embeds shape: [n_batch, ques_embed_dim]
        # output shape:  [n_batch, out_dim_modal_combine]
        # ----------------------------------------------------------------------
        modal_combine_model = build_modal_combine_module(final_modal_method,
                                                         par_pair.par,
                                                         img_embed_dim,
                                                         ques_embed_dim)
        # output is [N, out_dim_modal_combine]
        joint_embedding_dim = out_dim_modal_combine
        text_embeding_dim = 64
        image_embedding_dim  = 32
        # ----------------------------------------------------------------------
        # inputs:  joint_embeds
        # joint_embeds shape: [n_batch, joint_embedding_dim]
        # output shape:  [n_batch, num_ans_candidates]
        # ----------------------------------------------------------------------
        logit_classifier_model = logit_classifier(joint_embedding_dim,
                                                  num_ans_candidates,
                                                  img_hidden_dim=
                                                  image_embedding_dim,
                                                  txt_hidden_dim=
                                                  text_embeding_dim)

        vqa_model = vqa_multi_modal_model(img_embedding_model_list,
                                          ques_embedding_model_list,
                                          modal_combine_model,
                                          logit_classifier_model,
                                          img_encoding_model_list)

        vqa_txt_input = Variable(
            torch.randn(n_batch, question_len).type(
                torch.LongTensor) % num_vocab)

        vqa_img_input = Variable(torch.randn(n_batch,
                                             num_img_feats,
                                             in_dim))

        res = vqa_model(input_question_variable=vqa_txt_input,
                        image_feat_variables=[vqa_img_input],
                        image_dim_variable=None)
        self.assertEqual((n_batch, num_ans_candidates), res.shape)

    # def test_model(self):
    #     image_feat_dim = 40
    #     txt_embedding_dim = 300
    #     lstm_dim = 512
    #     hidden_size = 30
    #     num_of_loc = 5
    #     batch_size = 16
    #     num_vocab = 60
    #     num_ans_candidates = 35
    #     joint_embedding_dim = 500
    #     question_len = 13
    #     batch_first = True
    #     image_embedding_model = image_embedding(image_feat_dim,
    #                                             lstm_dim,
    #                                             hidden_size)
    #     question_embedding_model = QuestionEmbeding(num_vocab,
    #                                                 txt_embedding_dim,
    #                                                 lstm_dim,
    #                                                 lstm_layer=2,
    #                                                 dropout=0.1,
    #                                                 batch_first=batch_first)
    #     my_classifier = logit_classifier(joint_embedding_dim,
    #                                      num_ans_candidates,
    #                                      image_feat_dim,
    #                                      txt_embedding_dim)
    #     loss = torch.nn.CrossEntropyLoss()
    #
    #     my_model = top_down_bottom_up_model(image_embedding_model,
    #                                         question_embedding_model,
    #                                         my_classifier, loss)
    #     image_feat = np.random.rand(batch_size,
    #                                 num_of_loc,
    #                                 image_feat_dim)
    #     input_txt = Variable(torch.rand(batch_size,
    #                                     question_len).type(
    #         torch.LongTensor) % num_vocab)
    #     res = my_model(image_feat, input_txt, batch_first)
    #     self.assertEqual((batch_size, num_ans_candidates), res.shape)


if __name__ == '__main__':
    unittest.main()
