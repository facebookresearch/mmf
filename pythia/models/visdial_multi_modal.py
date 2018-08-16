import torch

from .top_down_bottom_up import VQAMultiModalModel
from pythia.modules.decoders import VisDialDiscriminator


class VisDialMultiModalModel(VQAMultiModalModel):
    def __init__(self, config):
        super(VisDialMultiModalModel, self).__init__(config)

    def build(self):
        self._init_question_embedding()
        self._init_image_encoders()
        self._init_image_embeddings()
        self._init_combine_layer()
        self._init_decoder()
        self._init_extras()

    def get_optimizer_parameters(self, config):
        # TODO: Update after implementing decoder
        params = [{'params': self.img_embeddings_list.parameters()},
                  {'params': self.question_embeddings.parameters()},
                  {'params': self.multi_modal_combine_layer.parameters()},
                  {'params': self.decoder.projection_layer.parameters()},
                  {'params': self.img_feat_encoders.parameters(),
                   'lr': (config['optimizer_attributes']['params']['lr']
                          * 0.1)}]

        return params

    def _update_question_embedding_args(self, args):
        parent = super(VisDialMultiModalModel, self)
        parent._update_question_embedding_args(args)
        # Add embedding vectors to args
        args['embedding_vectors'] = self.config['embedding_vectors']

    def _init_decoder(self):
        embedding = self.question_embeddings[0].module
        embedding_dim = self.question_embeddings[0].embedding_dim
        hidden_dim = self.multi_modal_combine_layer.out_dim

        self.decoder = VisDialDiscriminator({
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
        }, embedding)

    def forward(self,
                questions,
                answer_options,
                histories,
                image_features,
                image_dims,
                **kwargs):
            # TODO: Move flattening to separate function
            question_embeddings = []

            questions = questions.view(-1, questions.size(2))

            for idx, image_feature in enumerate(image_features):
                feature_size = image_feature.size()[2:]
                image_features[idx] = image_feature.view(-1, *feature_size)

            size = image_dims.size()[2:]
            image_dims = image_dims.view(-1, *size)

            for q_model in self.question_embeddings:
                q_embedding = q_model(questions)
                question_embeddings.append(q_embedding)
            question_embedding_total = torch.cat(question_embeddings, dim=1)

            assert (len(image_features) ==
                    len(self.img_feat_encoders)), \
                "number of image feature model doesnot equal \
                 to number of image features"

            image_embeddings = []

            for i, image_feat_variable in enumerate(image_features):
                image_dim_variable_use = None if i > 0 else image_dims
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
            logit_res = self.decoder(joint_embedding, {
                'answer_options': answer_options,
                'answer_options_len': kwargs['answer_options_len']
            })
            return logit_res
