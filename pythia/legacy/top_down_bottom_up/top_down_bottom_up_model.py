# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from global_variables.global_variables import use_cuda
from top_down_bottom_up.nonlinear_layer import nonlinear_layer


class top_down_bottom_up_model(nn.Module):
    def __init__(self, image_attention_model, question_embedding_models, classifier):
        super(top_down_bottom_up_model, self).__init__()
        self.image_attention_model = image_attention_model
        self.question_embedding_models = question_embedding_models
        self.classifier = classifier
        text_lstm_dim = sum([q.text_out_dim for q in question_embedding_models])
        joint_embedding_out_dim = classifier.input_dim
        image_feat_dim = image_attention_model.image_feat_dim
        self.nonLinear_question = nonlinear_layer(
            text_lstm_dim, joint_embedding_out_dim
        )
        self.nonLinear_image = nonlinear_layer(image_feat_dim, joint_embedding_out_dim)

    def forward(
        self, image_feat_variable, input_question_variable, input_answers=None, **kwargs
    ):

        question_embeddings = []
        for q_model in self.question_embedding_models:
            q_embedding = q_model(input_question_variable)
            question_embeddings.append(q_embedding)
        question_embedding = torch.cat(question_embeddings, dim=1)

        if isinstance(image_feat_variable, list):
            image_embeddings = []
            for idx, image_feat in enumerate(image_feat_variable):
                ques_embedding_each = torch.unsqueeze(question_embedding[idx, :], 0)
                image_feat_each = torch.unsqueeze(image_feat, dim=0)
                attention_each = self.image_attention_model(
                    image_feat_each, ques_embedding_each
                )
                image_embedding_each = torch.sum(attention_each * image_feat, dim=1)
                image_embeddings.append(image_embedding_each)
            image_embedding = torch.cat(image_embeddings, dim=0)
        else:
            attention = self.image_attention_model(
                image_feat_variable, question_embedding
            )
            image_embedding = torch.sum(attention * image_feat_variable, dim=1)

        joint_embedding = self.nonLinear_question(
            question_embedding
        ) * self.nonLinear_image(image_embedding)
        logit_res = self.classifier(joint_embedding)

        return logit_res


class vqa_multi_modal_model(nn.Module):
    def __init__(
        self,
        image_embedding_models_list,
        question_embedding_models,
        multi_modal_combine,
        classifier,
        image_feature_encode_list,
        inter_model=None,
    ):
        super(vqa_multi_modal_model, self).__init__()
        self.image_embedding_models_list = image_embedding_models_list
        self.question_embedding_models = question_embedding_models
        self.multi_modal_combine = multi_modal_combine
        self.classifier = classifier
        self.image_feature_encode_list = image_feature_encode_list
        self.inter_model = inter_model

    def forward(
        self,
        image_feat_variables,
        input_question_variable,
        image_dim_variable,
        input_answers=None,
        **kwargs
    ):
        question_embeddings = []
        for q_model in self.question_embedding_models:
            q_embedding = q_model(input_question_variable)
            question_embeddings.append(q_embedding)
        question_embedding_total = torch.cat(question_embeddings, dim=1)

        assert len(image_feat_variables) == len(
            self.image_feature_encode_list
        ), "number of image feature model doesnot equal \
             to number of image features"

        image_embeddings = []
        for i, image_feat_variable in enumerate(image_feat_variables):
            image_dim_variable_use = None if i > 0 else image_dim_variable
            image_feat_variable_ft = self.image_feature_encode_list[i](
                image_feat_variable
            )

            image_embedding_models_i = self.image_embedding_models_list[i]
            for i_model in image_embedding_models_i:
                i_embedding = i_model(
                    image_feat_variable_ft,
                    question_embedding_total,
                    image_dim_variable_use,
                )
                image_embeddings.append(i_embedding)

        image_embedding_total = torch.cat(image_embeddings, dim=1)

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.multi_modal_combine(
            image_embedding_total, question_embedding_total
        )
        logit_res = self.classifier(joint_embedding)

        return logit_res
