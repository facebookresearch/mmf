import torch

from torch import nn

from pythia.core.models.base_model import BaseModel
from pythia.core.registry import registry
from pythia.modules.embeddings import DefaultTextEmbedding
from pythia.modules.layers import (
        BCNet, FCNet, BiAttention, WeightNormClassifier
)


@registry.register_model("ban")
class BAN(BaseModel):
    def __init__(self, config):
        super(BAN, self).__init__(config)
        self.config = config

    def build(self):
        self._init_text_embedding()
        self._init_classifier()
        self._init_bilinear_attention()

    def _init_text_embedding(self):
        module_config = self.config['text_embedding']
        vocab_size = registry.get('vocabs.text_vocab').get_size()
        q_mod = DefaultTextEmbedding(module_config['num_hidden'],
                                     module_config['emb_size'],
                                     vocab_size,
                                     module_config['num_layers'],
                                     module_config['dropout'],
                                     module_config['bidirectional'],
                                     module_config['rnn_type'])
        self.q_emb = q_mod

    def _init_bilinear_attention(self):
        module_config = self.config['bilinear_attention']
        num_hidden = self.config['text_embedding']['num_hidden']
        v_dim = module_config['visual_feat_dim']

        v_att = BiAttention(v_dim,
                            num_hidden,
                            num_hidden,
                            module_config['gamma'])

        b_net = []
        q_prj = []

        for i in range(module_config['gamma']):
            b_net.append(BCNet(v_dim,
                               num_hidden,
                               num_hidden,
                               None,
                               k=module_config['bc_net']['k']))

            q_prj.append(FCNet(dims=[num_hidden, num_hidden],
                               act=module_config['fc_net']['activation'],
                               dropout=module_config['fc_net']['dropout']))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.v_att = v_att

    def _init_classifier(self):
        num_hidden = self.config['text_embedding']['num_hidden']
        out_dim = self.config['classifier']['out_dim']
        dropout = self.config['classifier']['dropout']
        self.classifier = WeightNormClassifier(num_hidden,
                                               out_dim,
                                               num_hidden * 2,
                                               dropout)

    def forward(self, image_features, texts, info={}, input_answers=None,
                **kwargs):
        v = image_features[0]
        q = texts

        q_emb = self.q_emb.forward_all(q)

        b_emb = [0] * self.config['bilinear_attention']['gamma']
        att, logits = self.v_att.forward_all(v, q_emb)

        for g in range(self.config['bilinear_attention']['gamma']):
            g_att = att[:, g, :, :]
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, g_att)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

        logits = self.classifier(q_emb.sum(1))

        return logits
