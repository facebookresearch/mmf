import torch

from torch import nn

from pythia.core.models.base_model import BaseModel
from pythia.core.registry import Registry
from pythia.modules.embeddings import ImageEmbedding
from pythia.modules.encoders import ImageEncoder
from pythia.modules.layers import ModalCombineLayer, ClassifierLayer, \
                                  ReLUWithWeightNormFC


@Registry.register_model("top_down_bottom_up")
class VQAMultiModalModel(BaseModel):
    def __init__(self, config):
        super(VQAMultiModalModel, self).__init__(config)
        self.config = config

    def build(self):
        self._init_text_embedding()
        self._init_feature_encoders("image")
        self._init_feature_embeddings("image")
        self._init_combine_layer("image", "text")
        self._init_classifier(self._get_classifier_input_dim())
        self._init_extras()

    def _update_text_embedding_args(self, args):
        # Add model_data_dir to kwargs
        args['model_data_dir'] = self.config['model_data_dir']

    def _init_feature_encoders(self, attr):
        feat_encoders = []
        feat_encoders_list_config = self.config[attr + '_feature_encodings']
        self.feat_dim = self.config[attr + '_feature_dim']
        setattr(self, attr + "_feature_dim", self.feat_dim)

        for feat_encoder in feat_encoders_list_config:
            encoder_type = feat_encoder['type']
            encoder_kwargs = feat_encoder['params']
            encoder_kwargs['model_data_dir'] = self.config['model_data_dir']
            feat_model = ImageEncoder(encoder_type, self.feat_dim,
                                      **encoder_kwargs)

            feat_encoders.append(feat_model)
            setattr(self, attr + "_feature_dim", feat_model.out_dim)

        setattr(self, attr + "_feature_encoders", nn.ModuleList(feat_encoders))

    def _init_feature_embeddings(self, attr):
        feature_embeddings_list = []
        num_feature_feat = self.config["num_" + attr + "_features"]

        self.feature_embeddings_out_dim = 0

        for _ in range(num_feature_feat):
            feature_embeddings = []
            feature_attn_model_list = self.config[attr + "_feature_embeddings"]

            for feature_attn_model_params in feature_attn_model_list:
                feature_embedding = ImageEmbedding(
                    getattr(self, attr + "_feature_dim"),
                    self.text_embeddings_out_dim,
                    **feature_attn_model_params
                )
                feature_embeddings.append(feature_embedding)
                self.feature_embeddings_out_dim += feature_embedding.out_dim

            feature_embeddings = nn.ModuleList(feature_embeddings)
            feature_embeddings_list.append(feature_embeddings)

        self.feature_embeddings_out_dim *= getattr(self, attr + "_feature_dim")

        setattr(self, attr + "_feature_embeddings_out_dim",
                self.feature_embeddings_out_dim)
        del self.feature_embeddings_out_dim
        setattr(self, attr + "_feature_embeddings_list",
                nn.ModuleList(feature_embeddings_list))

    def _get_embeddings_attr(self, attr):
        embedding_attr1 = attr
        if hasattr(self, attr + "_embeddings_out_dim"):
            embedding_attr1 = attr + "_embeddings_out_dim"
        else:
            embedding_attr1 = attr + "_feature_embeddings_out_dim"

        return embedding_attr1

    def _init_combine_layer(self, attr1, attr2):
        config_attr = attr1 + "_" + attr2 + "_modal_combine"

        multi_modal_combine_layer = ModalCombineLayer(
            self.config[config_attr]['type'],
            getattr(self, self._get_embeddings_attr(attr1)),
            getattr(self, self._get_embeddings_attr(attr2)),
            context_dim=getattr(self, self._get_embeddings_attr('context'),
                                None),
            **self.config[config_attr]['params']
        )

        setattr(self, attr1 + "_" + attr2 + "_multi_modal_combine_layer",
                multi_modal_combine_layer)

    def _init_classifier(self, combined_embedding_dim):
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
        combine_layer = self.image_text_multi_modal_combine_layer
        params = [{'params': self.image_feature_embeddings_list.parameters()},
                  {'params': self.text_embeddings.parameters()},
                  {'params': combine_layer.parameters()},
                  {'params': self.classifier.parameters()},
                  {'params': self.image_feature_encoders.parameters(),
                   'lr': (config['optimizer_attributes']['params']['lr']
                          * 0.1)}]

        return params

    def _get_classifier_input_dim(self):
        return self.image_text_multi_modal_combine_layer.out_dim

    def process_text_embedding(self, texts, embedding_attr='text_embeddings'):
        text_embeddings = []

        for t_model in getattr(self, embedding_attr):
            text_embedding = t_model(texts)
            text_embeddings.append(text_embedding)
        text_embeddding_total = torch.cat(text_embeddings, dim=1)
        return text_embeddding_total

    def process_feature_embedding(self, attr, feature_variables,
                                  feature_dim_variable, text_embedding_total):
        feature_embeddings = []
        feature_attentions = []

        if type(feature_variables) != list:
            feature_variables = [feature_variables]
        for i, feature_feat_variable in enumerate(feature_variables):
            feature_dim_variable_use = None if i > 0 else feature_dim_variable
            encoders_attr = attr + "_feature_encoders"
            feature_feat_variable_ft = (
                getattr(self, encoders_attr)[i](feature_feat_variable))

            list_attr = attr + "_feature_embeddings_list"
            feature_embedding_models_i = getattr(self, list_attr)[i]
            for i_model in feature_embedding_models_i:
                i_embedding, att = i_model(
                    feature_feat_variable_ft,
                    text_embedding_total, feature_dim_variable_use)
                feature_embeddings.append(i_embedding)
                feature_attentions.append(att)

        feature_embedding_total = torch.cat(feature_embeddings, dim=1)
        return feature_embedding_total, feature_attentions

    def combine_embeddings(self, *args):
        feature_names = args[0]
        feature_embeddings = args[1]

        layer = "_".join(feature_names) + "_multi_modal_combine_layer"
        return getattr(self, layer)(*feature_embeddings)

    def calculate_logits(self, joint_embedding, **kwargs):
        return self.classifier(joint_embedding)

    def forward(self,
                image_features,
                texts,
                info={},
                input_answers=None, **kwargs):

        input_text_variable = texts
        image_dim_variable = info.get('image_dim', None)
        image_feature_variables = image_features
        text_embedding_total = self.process_text_embedding(input_text_variable)

        assert (len(image_feature_variables) ==
                len(self.image_feature_encoders)), \
            "number of image feature model doesnot equal \
             to number of image features"

        image_embedding_total, _ = self.process_feature_embedding(
            "image",
            image_feature_variables,
            image_dim_variable,
            text_embedding_total
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(["image", "text"],
                                                  [image_embedding_total,
                                                  text_embedding_total])

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
