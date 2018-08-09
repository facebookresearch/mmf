# TODO: Update kwargs with defaults
import torch
import os
import pickle
import numpy as np

from torch import nn

from pythia.modules.attention import AttentionLayer


class QuestionEmbedding(nn.Module):
    def __init__(self, emb_type, **kwargs):
        super(QuestionEmbedding, self).__init__()
        self.data_root_dir = kwargs.get('data_root_dir', None)

        # Update kwargs here
        if emb_type == "default":
            params = {
                'hidden_dim': kwargs['hidden_dim'],
                'embedding_dim': kwargs['embedding_dim'],
                'vocab_size': kwargs['vocab_size'],
                'num_layers': kwargs['num_layers'],
                'dropout': kwargs['dropout']
            }
            embedding = DefaultQuestionEmbedding(**params)
            self.init_embedding(embedding.embedding,
                                kwargs.get('embedding_init', None),
                                kwargs.get('embedding_init_file', None))
            self.module = embedding

        elif emb_type == "attention":
            embedding = AttentionQuestionEmbedding(**kwargs)
            self.init_embedding(embedding.embedding,
                                kwargs.get('embedding_init', None),
                                kwargs.get('embedding_init_file', None))
            self.module = embedding
        else:
            raise NotImplementedError("Unknown question embedding '%s'"
                                      % emb_type)

        self.text_out_dim = self.module.text_out_dim

    def init_embedding(self, embedding, embedding_init, embedding_init_file):
        if embedding_init is not None:
            weights = torch.from_numpy(embedding_init)
            embedding.weights.data.copy_(weights)

        if embedding_init_file is not None and self.data_root_dir is not None:
            embedding_file = embedding_init_file
            if not os.path.isabs(embedding_file):
                embedding_file = os.path.join(self.data_root_dir,
                                              embedding_file)
            embedding_init = np.load(embedding_file)
            embedding.weight.data.copy_(torch.from_numpy(embedding_init))

        return embedding

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class DefaultQuestionEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim,
                 vocab_size, num_layers, dropout):
        super(DefaultQuestionEmbedding, self).__init__()
        self.text_out_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.recurrent_encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        embedded_x = self.embedding(x)
        out, _ = self.recurrent_encoder(embedded_x)
        # Return last state
        return out[:, -1]


class AttentionQuestionEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim,
                 vocab_size, num_layers, dropout, **kwargs):
        super(AttentionQuestionEmbedding, self).__init__()

        self.text_out_dim = hidden_dim * kwargs['conv2_out']

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.recurrent_unit = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(p=dropout)

        conv1_out = kwargs['conv1_out']
        conv2_out = kwargs['conv2_out']
        kernel_size = kwargs['kernel_size']
        padding = kwargs['padding']

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _ = x.data.shape
        embedded_x = self.embedding(x)  # N * T * embedding_dim

        # self.recurrent_unit.flatten_parameters()
        lstm_out, _ = self.recurrent_unit(embedded_x)  # N * T * hidden_dim
        lstm_drop = self.dropout(lstm_out)  # N * T * hidden_dim
        lstm_reshape = lstm_drop.permute(0, 2, 1)  # N * hidden_dim * T

        qatt_conv1 = self.conv1(lstm_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        return qtt_feature_concat


class ImageEmbedding(nn.Module):
    '''
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    question_embedding:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]


    '''
    def __init__(self, img_dim, question_dim, **kwargs):
        super(ImageEmbedding, self).__init__()

        self.image_attention_model = AttentionLayer(
            img_dim,
            question_dim,
            **kwargs
        )
        self.out_dim = self.image_attention_model.out_dim

    def forward(self, image_feat_variable, question_embedding, image_dims):
        # N x K x n_att
        attention = self.image_attention_model(
            image_feat_variable, question_embedding, image_dims)
        att_reshape = attention.permute(0, 2, 1)
        tmp_embedding = torch.bmm(
            att_reshape, image_feat_variable)  # N x n_att x image_dim
        batch_size = att_reshape.size(0)
        image_embedding = tmp_embedding.view(batch_size, -1)

        return image_embedding


class ImageFinetune(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file):
        super(ImageFinetune, self).__init__()
        with open(weights_file, 'rb') as w:
            weights = pickle.load(w)
        with open(bias_file, 'rb') as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3
