import torch

from torch import nn

from .top_down_bottom_up import VQAMultiModalModel


class VisDialMultiModalModel(VQAMultiModalModel):
    def __init__(self, config):
        super(VisDialMultiModalModel, self).__init__(config)

    def _init_layer(self):
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
                  {'params': self.decoder.parameters()},
                  {'params': self.img_feat_encoders.parameters(),
                   'lr': (config['optimizer_attributes']['params']['lr']
                          * 0.1)}]

        return params
