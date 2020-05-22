# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.models.pythia import Pythia
from mmf.modules.decoders import VisDialDiscriminator


class VisDialMultiModalModel(Pythia):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        self._init_text_embedding()
        self._init_image_encoders()
        self._init_image_embeddings()
        self._init_combine_layer()
        self._init_decoder()
        self._init_extras()

    def _init_text_embedding(self):
        parent = super()
        parent._init_text_embedding("text_embeddings", False)
        parent._init_text_embedding("history_embeddings", True)

    def get_optimizer_parameters(self, config):
        # TODO: Update after implementing decoder
        params = [
            {"params": self.img_embeddings_list.parameters()},
            {"params": self.text_embeddings.parameters()},
            {"params": self.multi_modal_combine_layer.parameters()},
            {"params": self.decoder.projection_layer.parameters()},
            {
                "params": self.img_feat_encoders.parameters(),
                "lr": (config.optimizer.params.lr * 0.1),
            },
        ]

        return params

    def _update_text_embedding_args(self, args):
        parent = super()
        parent._update_text_embedding_args(args)
        # Add embedding vectors to args
        args.embedding_vectors = self.config.embedding_vectors

    def _init_decoder(self):
        embedding = self.text_embeddings[0].module
        embedding_dim = self.text_embeddings[0].embedding_dim
        hidden_dim = self.multi_modal_combine_layer.out_dim

        self.decoder = VisDialDiscriminator(
            {"embedding_dim": embedding_dim, "hidden_dim": hidden_dim}, embedding
        )

    def combine_embeddings(self, *args):
        return self.multi_modal_combine_layer(*args)

    def calculate_logits(self, joint_embedding, **kwargs):
        return self.decoder(joint_embedding, kwargs)

    def forward(
        self, texts, answer_options, histories, image_features, image_dims, **kwargs
    ):

        texts = texts.view(-1, texts.size(2))
        histories = histories.view(-1, histories.size(2))
        text_embedding_total = self.process_text_embedding(texts)
        histories_total = self.process_text_embedding(histories, "history_embeddings")

        for idx, image_feature in enumerate(image_features):
            feature_size = image_feature.size()[2:]
            image_features[idx] = image_feature.view(-1, *feature_size)

        size = image_dims.size()[2:]
        image_dims = image_dims.view(-1, *size)

        assert len(image_features) == len(
            self.img_feat_encoders
        ), "number of image feature model doesnot equal \
                 to number of image features"

        image_embedding_total = self.process_image_embedding(
            image_features, image_dims, text_embedding_total
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(
            image_embedding_total, text_embedding_total, histories_total
        )

        decoder_info = {
            "answer_options": answer_options,
            "answer_options_len": kwargs["answer_options_len"],
        }
        return self.calculate_logits(joint_embedding, **decoder_info)
