import torch

from torch import nn

from pythia.modules.embeddings import QuestionEmbedding, ImageEmbedding
from pythia.modules.encoders import ImageEncoder
from pythia.modules.layers import ModalCombineLayer, ClassifierLayer, GatedTanh


class VQAMultiModalModel(nn.Module):
    def __init__(self, config):
        super(VQAMultiModalModel, self).__init__()
        self.config = config
        self.num_choices = self.config['num_choices']

        self._init_layers()

    def _init_layers(self):
        self._init_question_embedding()
        self._init_image_encoders()
        self._init_image_embeddings()
        self._init_combine_layer()
        self._init_classifier()

    def _init_question_embedding(self):
        question_embeddings = []
        question_embeddings_list_config = self.config['question_embeddings']

        self.question_embeddings_out_dim = 0

        for question_embedding in question_embeddings_list_config:
            embedding_type = question_embedding['embedding_type']
            embedding_kwargs = question_embeddings['params']

            embedding = QuestionEmbedding(embedding_type, embedding_kwargs)
            question_embeddings.append(embedding)
            self.question_embeddings_out_dim += embedding.text_out_dim

        self.question_embeddings = nn.ModuleList(question_embeddings)

    def _init_image_encoders(self):
        img_feat_encoders = []
        img_feat_encoders_list_config = self.config['image_feature_encodings']
        self.img_feat_dim = self.config['image_feature_dim']

        for img_feat_encoder in img_feat_encoders_list_config:
            encoder_type = img_feat_encoder['encoder_type']
            encoder_kwargs = img_feat_encoder['params']
            img_feat_model = ImageEncoder(encoder_type, self.img_feat_dim,
                                          encoder_kwargs)

            img_feat_encoders.append(img_feat_model)
            self.img_feat_dim = img_feat_model.out_dim

        self.img_feat_encoders = nn.ModuleList(img_feat_encoders)

    def _init_image_embeddings(self):
        img_embeddings_list = []
        num_img_feat = self.config['num_image_features']

        self.img_embeddings_out_dim = 0

        for _ in range(num_img_feat):
            img_embeddings = []
            img_attn_model_list = self.config['image_embeddings']

            for img_attn_model_params in img_attn_model_list:
                img_embedding = ImageEmbedding(
                    self.img_feat_dim,
                    self.question_embeddings_out_dim,
                    img_attn_model_params
                )
                img_embeddings.append(img_embedding)
                self.img_embeddings_out_dim += img_embedding.out_dim

            img_embeddings = nn.ModuleList(img_embeddings)
            img_embeddings_list.append(img_embeddings)

        self.img_embeddings_out_dim *= self.img_feat_dim
        self.img_embeddings_list = nn.ModuleList(img_embeddings_list)

    def _init_combine_layer(self):
        self.multi_modal_combine_layer = ModalCombineLayer(
            self.config['modal_combine']['combine_type'],
            self.img_embeddings_out_dim,
            self.question_embeddings_out_dim,
            self.config['modal_combine']['params']
        )

    def _init_classifier(self):
        combined_embedding_dim = self.multi_modal_combine_layer.out_dim

        self.classifier = ClassifierLayer(
            self.config['classifier']['classifier_type'],
            in_dim=combined_embedding_dim,
            out_dim=self.num_choices
        )

    def forward(self,
                image_feat_variables,
                input_question_variable,
                image_dim_variable,
                input_answers=None, **kwargs):
        question_embeddings = []
        for q_model in self.question_embeddings:
            q_embedding = q_model(input_question_variable)
            question_embeddings.append(q_embedding)
        question_embedding_total = torch.cat(question_embeddings, dim=1)

        assert (len(image_feat_variables) ==
                len(self.img_feat_encoders)), \
            "number of image feature model doesnot equal \
             to number of image features"

        image_embeddings = []

        for i, image_feat_variable in enumerate(image_feat_variables):
            image_dim_variable_use = None if i > 0 else image_dim_variable
            image_feat_variable_ft = (
                self.img_feat_encoders[i](image_feat_variable))

            image_embedding_models_i = self.img_embeddings_list[i]
            for i_model in image_embedding_models_i:
                i_embedding = i_model(
                    image_feat_variable_ft,
                    question_embedding_total, image_dim_variable_use)
                image_embeddings.append(i_embedding)

        image_embedding_total = torch.cat(image_embeddings, dim=1)

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.multi_modal_combine_layer(
            image_embedding_total, question_embedding_total)
        logit_res = self.classifier(joint_embedding)

        return logit_res


class TopDownBottomUpModel(nn.Module):
    def __init__(self, image_attention_model,
                 question_embedding_models, classifier):
        super(TopDownBottomUpModel, self).__init__()
        self.image_attention_model = image_attention_model
        self.question_embedding_models = question_embedding_models
        self.classifier = classifier
        text_lstm_dim = sum(
            [q.text_out_dim for q in question_embedding_models])
        joint_embedding_out_dim = classifier.input_dim
        image_feat_dim = image_attention_model.image_feat_dim
        self.non_linear_question = GatedTanh(
            text_lstm_dim, joint_embedding_out_dim)
        self.non_linear_image = GatedTanh(
            image_feat_dim, joint_embedding_out_dim)

    def forward(self, image_feat_variable,
                input_question_variable, input_answers=None, **kwargs):
        question_embeddings = []
        for q_model in self.question_embedding_models:
            q_embedding = q_model(input_question_variable)
            question_embeddings.append(q_embedding)
        question_embedding = torch.cat(question_embeddings, dim=1)

        if isinstance(image_feat_variable, list):
            image_embeddings = []
            for idx, image_feat in enumerate(image_feat_variable):
                ques_embedding_each = torch.unsqueeze(
                    question_embedding[idx, :], 0)
                image_feat_each = torch.unsqueeze(image_feat, dim=0)
                attention_each = self.image_attention_model(
                    image_feat_each, ques_embedding_each)
                image_embedding_each = torch.sum(
                    attention_each * image_feat, dim=1)
                image_embeddings.append(image_embedding_each)
            image_embedding = torch.cat(image_embeddings, dim=0)
        else:
            attention = self.image_attention_model(
                image_feat_variable, question_embedding)
            image_embedding = torch.sum(attention * image_feat_variable, dim=1)

        joint_embedding = self.non_linear_question(
            question_embedding) * self.non_linear_image(image_embedding)
        logit_res = self.classifier(joint_embedding)

        return logit_res
