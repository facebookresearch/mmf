import torch

from torch import nn

from pythia.modules.embeddings import TextEmbedding, ImageEmbedding
from pythia.modules.encoders import ImageEncoder
from pythia.modules.layers import ModalCombineLayer, ClassifierLayer, \
                                  ReLUWithWeightNormFC


class VQAMultiModalModel(nn.Module):
    def __init__(self, config):
        super(VQAMultiModalModel, self).__init__()
        self.config = config

    def build(self):
        self._init_text_embedding()
        self._init_image_encoders()
        self._init_image_embeddings()
        self._init_combine_layer()
        self._init_classifier()
        self._init_extras()

    def _init_text_embedding(self, attr='text_embeddings',
                             bidirectional=False):
        text_embeddings = []
        text_embeddings_list_config = self.config['text_embeddings']

        self.text_embeddings_out_dim = 0

        for text_embedding in text_embeddings_list_config:
            embedding_type = text_embedding['type']
            embedding_kwargs = text_embedding['params']
            embedding_kwargs['bidirectional'] = bidirectional
            self._update_text_embedding_args(embedding_kwargs)

            embedding = TextEmbedding(embedding_type, **embedding_kwargs)
            text_embeddings.append(embedding)
            self.text_embeddings_out_dim += embedding.text_out_dim

        setattr(self, attr, nn.ModuleList(text_embeddings))

    def _update_text_embedding_args(self, args):
        # Add data_root_dir to kwargs
        args['data_root_dir'] = self.config['data_root_dir']
        args['vocab_size'] = self.config['vocab_size']

    def _init_image_encoders(self):
        img_feat_encoders = []
        img_feat_encoders_list_config = self.config['image_feature_encodings']
        self.img_feat_dim = self.config['image_feature_dim']

        for img_feat_encoder in img_feat_encoders_list_config:
            encoder_type = img_feat_encoder['type']
            encoder_kwargs = img_feat_encoder['params']
            encoder_kwargs['data_root_dir'] = self.config['data_root_dir']
            img_feat_model = ImageEncoder(encoder_type, self.img_feat_dim,
                                          **encoder_kwargs)

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
                    self.text_embeddings_out_dim,
                    **img_attn_model_params
                )
                img_embeddings.append(img_embedding)
                self.img_embeddings_out_dim += img_embedding.out_dim

            img_embeddings = nn.ModuleList(img_embeddings)
            img_embeddings_list.append(img_embeddings)

        self.img_embeddings_out_dim *= self.img_feat_dim
        self.img_embeddings_list = nn.ModuleList(img_embeddings_list)

    def _init_combine_layer(self):
        self.multi_modal_combine_layer = ModalCombineLayer(
            self.config['modal_combine']['type'],
            self.img_embeddings_out_dim,
            self.text_embeddings_out_dim,
            **self.config['modal_combine']['params']
        )

    def _init_classifier(self):
        combined_embedding_dim = self.multi_modal_combine_layer.out_dim
        num_choices = self.config['num_choices']

        self.classifier = ClassifierLayer(
            self.config['classifier']['type'],
            in_dim=combined_embedding_dim,
            out_dim=num_choices,
            **self.config['classifier']['params']
        )

    def _init_extras(self):
        self.inter_model = None

    def get_optimizer_parameters(self, config):
        params = [{'params': self.img_embeddings_list.parameters()},
                  {'params': self.text_embeddings.parameters()},
                  {'params': self.multi_modal_combine_layer.parameters()},
                  {'params': self.classifier.parameters()},
                  {'params': self.img_feat_encoders.parameters(),
                   'lr': (config['optimizer_attributes']['params']['lr']
                          * 0.1)}]

        return params

    def process_text_embedding(self, texts, embedding_attr='text_embeddings'):
        text_embeddings = []

        for t_model in getattr(self, embedding_attr):
            text_embedding = t_model(texts)
            text_embeddings.append(text_embedding)
        text_embeddding_total = torch.cat(text_embeddings, dim=1)
        return text_embeddding_total

    def process_image_embedding(self, image_feature_variables,
                                image_dim_variable, text_embedding_total):
        image_embeddings = []

        for i, image_feat_variable in enumerate(image_feature_variables):
            image_dim_variable_use = None if i > 0 else image_dim_variable
            image_feat_variable_ft = (
                self.img_feat_encoders[i](image_feat_variable))

            image_embedding_models_i = self.img_embeddings_list[i]
            for i_model in image_embedding_models_i:
                i_embedding = i_model(
                    image_feat_variable_ft,
                    text_embedding_total, image_dim_variable_use)
                image_embeddings.append(i_embedding)

        image_embedding_total = torch.cat(image_embeddings, dim=1)
        return image_embedding_total

    def combine_embeddings(self, *args):
        image_embedding = args[0]
        text_embedding = args[1]

        return self.multi_modal_combine_layer(image_embedding, text_embedding)

    def calculate_logits(self, joint_embedding, **kwargs):
        return self.classifier(joint_embedding)

    def forward(self,
                image_features,
                texts,
                image_dim,
                input_answers=None, **kwargs):

        input_text_variable = texts
        image_dim_variable = image_dim
        image_feature_variables = image_features
        text_embedding_total = self.process_text_embedding(input_text_variable)

        assert (len(image_feature_variables) ==
                len(self.img_feat_encoders)), \
            "number of image feature model doesnot equal \
             to number of image features"

        image_embedding_total = self.process_image_embedding(
            image_feature_variables,
            image_dim_variable,
            text_embedding_total
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(image_embedding_total,
                                                  text_embedding_total)

        return self.calculate_logits(joint_embedding)


class TopDownBottomUpModel(nn.Module):
    def __init__(self, image_attention_model,
                 text_embedding_models, classifier):
        super(TopDownBottomUpModel, self).__init__()
        self.image_attention_model = image_attention_model
        self.text_embedding_models = text_embedding_models
        self.classifier = classifier
        text_lstm_dim = sum(
            [q.text_out_dim for q in text_embedding_models])
        joint_embedding_out_dim = classifier.input_dim
        image_feat_dim = image_attention_model.image_feat_dim
        self.non_linear_text = ReLUWithWeightNormFC(
            text_lstm_dim, joint_embedding_out_dim)
        self.non_linear_image = ReLUWithWeightNormFC(
            image_feat_dim, joint_embedding_out_dim)

    def forward(self, image_feat_variable,
                input_text_variable, input_answers=None, **kwargs):
        text_embeddings = []
        for q_model in self.text_embedding_models:
            q_embedding = q_model(input_text_variable)
            text_embeddings.append(q_embedding)
        text_embedding = torch.cat(text_embeddings, dim=1)

        if isinstance(image_feat_variable, list):
            image_embeddings = []
            for idx, image_feat in enumerate(image_feat_variable):
                ques_embedding_each = torch.unsqueeze(
                    text_embedding[idx, :], 0)
                image_feat_each = torch.unsqueeze(image_feat, dim=0)
                attention_each = self.image_attention_model(
                    image_feat_each, ques_embedding_each)
                image_embedding_each = torch.sum(
                    attention_each * image_feat, dim=1)
                image_embeddings.append(image_embedding_each)
            image_embedding = torch.cat(image_embeddings, dim=0)
        else:
            attention = self.image_attention_model(
                image_feat_variable, text_embedding)
            image_embedding = torch.sum(attention * image_feat_variable, dim=1)

        joint_embedding = self.non_linear_text(
            text_embedding) * self.non_linear_image(image_embedding)
        logit_res = self.classifier(joint_embedding)

        return logit_res
