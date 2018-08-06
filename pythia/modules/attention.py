import torch

from torch import nn
from .layers import GatedTanh, ModalCombineLayer, TransformLayer


class AttentionLayer(nn.Module):
    def __init__(self, image_dim, question_dim, **kwargs):
        super(AttentionLayer, self).__init__()

        combine_type = kwargs['modal_combine']['type']
        combine_params = kwargs['modal_combine']['params']
        modal_combine_layer = ModalCombineLayer(combine_type, image_dim,
                                                question_dim, **combine_params)

        transform_type = kwargs['transform']['type']
        transform_params = kwargs['transform']['params']
        transform_layer = TransformLayer(transform_type,
                                         modal_combine_layer.out_dim,
                                         **transform_params)

        normalization = kwargs['normalization']

        self.module = TopDownAttention(modal_combine_layer, transform_layer,
                                       normalization)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ConcatenationAttention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size):
        super(ConcatenationAttention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.fa = GatedTanh(image_feat_dim + txt_rnn_embeding_dim,
                            hidden_size)
        self.lc = nn.Linear(hidden_size, 1)

    def forward(self, image_feat, question_embedding):
        _, num_location, _ = image_feat.shape
        question_embedding_expand = torch.unsqueeze(
            question_embedding, 1).expand(-1, num_location, -1)
        concat_feature = torch.cat(
            (image_feat, question_embedding_expand), dim=2)
        raw_attention = self.lc(self.fa(concat_feature))
        # softmax across locations
        attention_weights = nn.functional.softmax(raw_attention, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class ProjectAttention(nn.Module):
    def __init__(self,
                 image_feat_dim,
                 txt_rnn_embeding_dim,
                 hidden_size,
                 dropout=0.2):
        super(ProjectAttention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.fa_image = GatedTanh(image_feat_dim, hidden_size)
        self.fa_txt = GatedTanh(txt_rnn_embeding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lc = nn.Linear(hidden_size, 1)

    def compute_raw_att(self, image_feat, question_embedding):
        num_location = image_feat.shape[1]
        image_fa = self.fa_image(image_feat)
        question_fa = self.fa_txt(question_embedding)
        question_fa_expand = torch.unsqueeze(
            question_fa, 1).expand(-1, num_location, -1)
        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)
        raw_attention = self.lc(joint_feature)
        return raw_attention

    def forward(self, image_feat, question_embedding):
        raw_attention = self.compute_raw_att(image_feat, question_embedding)
        # softmax across locations
        attention_weights = nn.functional.softmax(raw_attention, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class DoubleProjectAttention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim,
                 hidden_size, dropout=0.2):
        super(DoubleProjectAttention, self).__init__()
        self.att1 = ProjectAttention(image_feat_dim, txt_rnn_embeding_dim,
                                     hidden_size, dropout)
        self.att2 = ProjectAttention(image_feat_dim, txt_rnn_embeding_dim,
                                     hidden_size, dropout)
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim

    def forward(self, image_feat, question_embedding):
        att1 = self.att1.compute_raw_att(image_feat, question_embedding)
        att2 = self.att2.compute_raw_att(image_feat, question_embedding)
        raw_attn_weights = att1 + att2
        # softmax across locations
        attention_weights = nn.functional.softmax(raw_attn_weights, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class TopDownAttention(nn.Module):
    def __init__(self, combination_layer, transform_module, normalization):
        super(TopDownAttention, self).__init__()
        self.combination_layer = combination_layer
        self.normalization = normalization
        self.transform = transform_module
        self.out_dim = self.transform.out_dim

    @staticmethod
    def _mask_attentions(attention, image_locs):
        batch_size, num_loc, n_att = attention.data.shape
        tmp1 = torch.unsqueeze(
            torch.arange(0, num_loc).type(torch.LongTensor),
            dim=0).expand(batch_size, num_loc)
        use_cuda = attention.is_cuda
        tmp1 = tmp1.cuda() if use_cuda else tmp1
        tmp2 = torch.unsqueeze(image_locs.data, 1).expand(batch_size, num_loc)
        mask = torch.ge(tmp1, tmp2)
        mask = torch.unsqueeze(mask, 2).expand_as(attention)
        attention.data.masked_fill_(mask, 0)
        return attention

    def forward(self, image_feat, question_embedding, image_locs=None):
        # N x K x joint_dim
        joint_feature = self.combination_layer(image_feat, question_embedding)
        # N x K x n_att
        raw_attn = self.transform(joint_feature)

        if self.normalization.lower() == 'softmax':
            attention = nn.functional.softmax(raw_attn, dim=1)
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)
                masked_attention_sum = torch.sum(masked_attention,
                                                 dim=1, keepdim=True)
                masked_attention = masked_attention / masked_attention_sum
            else:
                masked_attention = attention

        elif self.normalization.lower() == 'sigmoid':
            attention = nn.functional.sigmoid(raw_attn)
            masked_attention = attention
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)

        return masked_attention
