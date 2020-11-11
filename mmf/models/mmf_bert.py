# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.common.registry import registry
from mmf.models.pythia import Pythia
from mmf.modules.embeddings import ProjectionEmbedding
from mmf.utils.transform import transform_to_batch_sequence
from torch import nn
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertForPreTraining,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainingHeads,
)


@registry.register_model("mmf_bert")
class MMFBert(Pythia):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/mmf_bert/defaults.yaml"

    def build(self):
        super().build()
        self.tie_weights()

        if getattr(self.config, "freeze_base", False):
            for n, p in self.named_parameters():
                if "classifier" not in n:
                    p.requires_grad = False

    def _build_word_embedding(self):
        self.bert_config = BertConfig.from_pretrained(self.config.bert_model_name)
        if self.config.pretrained_bert:
            bert_model = BertForPreTraining.from_pretrained(self.config.bert_model_name)
            self.word_embedding = bert_model.bert.embeddings
            self.pooler = bert_model.bert.pooler
            self.pooler.apply(self.init_weights)

        else:
            self.pooler = BertPooler(self.bert_config)
            self.word_embedding = BertEmbeddings(self.bert_config)

    def _init_classifier(self, hidden_size):
        if "pretraining" in self.config.training_head_type:
            self.classifier = BertPreTrainingHeads(self.bert_config)
        if "vqa" in self.config.training_head_type:
            self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
            self.answer_space_size = 3129
            self.classifier = nn.Sequential(
                BertPredictionHeadTransform(self.bert_config),
                nn.Linear(self.bert_config.hidden_size, self.answer_space_size),
            )
            # self.classifier = nn.Linear(self.bert_config.hidden_size,
            # self.answer_space_size)
        elif "vizwiz" in self.config.training_head_type:
            self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
            self.answer_space_size = 7371
            self.classifier = nn.Sequential(
                BertPredictionHeadTransform(self.bert_config),
                nn.Linear(self.bert_config.hidden_size, self.answer_space_size),
            )
            # self.classifier = nn.Linear(self.bert_config.hidden_size,
            # self.answer_space_size)
        elif self.config.training_head_type == "visual_entailment":
            self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
            self.classifier = nn.Sequential(
                BertPredictionHeadTransform(self.bert_config),
                nn.Linear(self.bert_config.hidden_size, 3),
            )
            # self.classifier = nn.Linear(self.bert_config.hidden_size, 3)

    def _init_text_embeddings(self, attr="text"):
        self.text_embeddings_out_dim = self.bert_config.hidden_size
        self.text_embedding = nn.MultiheadAttention(**self.config.text_embeddings[0])

    def _tie_or_clone_weights(self, first_module, second_module):
        """Tie or clone module weights depending of weither we are using
        TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """Make sure we are sharing the input and output embeddings.
        Export to TorchScript can't handle parameter sharing so we are cloning
        them instead.
        """
        if hasattr(self, "cls"):
            self._tie_or_clone_weights(
                self.cls.predictions.decoder, self.word_embeddings.word_embeddings
            )

    def _init_feature_embeddings(self, attr):
        feature_embeddings_list = []
        num_feature_feat = len(getattr(self.config, f"{attr}_feature_encodings"))

        self.image_feature_projection = ProjectionEmbedding(
            **self.config.image_feature_projection
        )
        self.feature_embeddings_out_dim = 0

        if self.config.image_intra_attention:
            self.image_feature_intra_attention = nn.MultiheadAttention(
                **self.config.image_feature_attentions[0]
            )

        for _ in range(num_feature_feat):
            feature_embeddings = []
            feature_attn_model_list = self.config[attr + "_feature_embeddings"]

            for feature_attn_model_params in feature_attn_model_list:
                feature_embedding = nn.MultiheadAttention(**feature_attn_model_params)
                feature_embeddings.append(feature_embedding)
                self.feature_embeddings_out_dim += feature_attn_model_params[
                    "embed_dim"
                ]

            feature_embeddings = nn.ModuleList(feature_embeddings)
            feature_embeddings_list.append(feature_embeddings)

        setattr(
            self, attr + "_feature_embeddings_out_dim", self.feature_embeddings_out_dim
        )
        del self.feature_embeddings_out_dim
        setattr(
            self,
            attr + "_feature_embeddings_list",
            nn.ModuleList(feature_embeddings_list),
        )

    def get_optimizer_parameters(self, config):
        param_optimizer = list(self.named_parameters())
        image_feature_encoders_params = [
            n for n in param_optimizer if "image_feature_encoders" in n[0]
        ]
        param_optimizer = [
            n for n in param_optimizer if "image_feature_encoders" not in n[0]
        ]

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [p for _, p in image_feature_encoders_params],
                "lr": (config.optimizer.params.lr * 0.1),
                "weight_decay": 0.01,
            },
        ]

        return optimizer_grouped_parameters

    # WARNING(ASG): This doesn't have finetune_lr_multiplier option enabled yet

    def process_text_embedding(self, text_embedding, key_padding_mask=None):
        text_embedding = text_embedding.transpose(0, 1)
        embedding, _ = self.text_embedding(
            text_embedding,
            text_embedding,
            text_embedding,
            key_padding_mask=key_padding_mask,
        )

        return embedding.transpose(0, 1)

    def process_feature_embedding(
        self,
        attr,
        sample_list,
        text_embedding_total,
        key_padding_mask=None,
        attn_mask=None,
        extra=None,
        batch_size_t=None,
    ):
        if extra is None:
            extra = []
        feature_embeddings = []
        feature_attentions = []
        features = []
        batch_size_t = (
            sample_list.get_batch_size() if batch_size_t is None else batch_size_t
        )

        # Convert list of keys to the actual values
        extra = sample_list.get_fields(extra)

        feature_idx = 0

        # Get all of the features, which are in the form, "image_feature_0"
        # "image_feature_1" ...
        while True:
            feature = getattr(sample_list, f"{attr}_feature_{feature_idx:d}", None)
            if feature is None:
                break
            feature_idx += 1
            feature = feature[:batch_size_t]
            features.append(feature)

        feature_encoders = getattr(self, attr + "_feature_encoders")
        # Each feature should have a separate image feature encoders
        assert len(features) == len(feature_encoders), (
            "Number of feature encoders, {} are not equal "
            "to number of features, {}.".format(len(feature_encoders), len(features))
        )

        # Now, iterate to get final attended image features
        for i, feature in enumerate(features):
            # Get info related to the current feature. info is generally
            # in key of format "image_info_0" for 0th feature
            feature_info = getattr(sample_list, f"{attr}_info_{i:d}", {})
            # For Pythia, we need max_features to mask attention
            feature_dim = getattr(feature_info, "max_features", None)
            if feature_dim is not None:
                feature_dim = feature_dim[:batch_size_t]

            # Attribute in which encoders are saved, for "image" it
            # will be "image_feature_encoders", other example is
            # "context_feature_encoders"
            encoders_attr = attr + "_feature_encoders"
            feature_encoder = getattr(self, encoders_attr)[i]

            # Encode the features
            encoded_feature = feature_encoder(feature)

            # Get all of the feature embeddings
            list_attr = attr + "_feature_embeddings_list"
            feature_embedding_models = getattr(self, list_attr)[i]
            encoded_feature = self.image_feature_projection(encoded_feature)
            encoded_feature = encoded_feature.transpose(0, 1)
            text_embedding_total = text_embedding_total.transpose(0, 1)

            if self.config.image_intra_attention:
                encoded_feature, _ = self.image_feature_intra_attention(
                    encoded_feature,
                    encoded_feature,
                    encoded_feature,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            # Forward through these embeddings one by one
            for feature_embedding_model in feature_embedding_models:
                inp = (text_embedding_total, encoded_feature, encoded_feature)
                embedding, attention = feature_embedding_model(
                    *inp, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
                feature_embeddings.append(embedding.transpose(0, 1))
                feature_attentions.append(attention.squeeze(-1))

        # Concatenate all features embeddings and return along with attention
        feature_embedding_total = torch.cat(feature_embeddings, dim=1)
        return feature_embedding_total, feature_attentions

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal
            # for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, sample_list):
        # bert text input
        input_ids = sample_list.input_ids
        input_mask = sample_list.input_mask
        input_type_ids = sample_list.segment_ids
        input_ids = transform_to_batch_sequence(input_ids)
        input_type_ids = transform_to_batch_sequence(input_type_ids)
        input_mask = transform_to_batch_sequence(input_mask)

        if input_mask is None:
            input_mask = torch.ones_like(input_ids)
        if input_type_ids is None:
            input_type_ids = torch.zeros_like(input_ids)
        attention_mask = input_mask.unsqueeze(1).unsqueeze(2)

        # pretraining labels
        masked_lm_labels = getattr(sample_list, "lm_label_ids", None)
        masked_lm_labels = transform_to_batch_sequence(masked_lm_labels)
        # pretraining labels
        # is_random_next = getattr(sample_list, "is_correct", None)
        # TODO(aps): Fix later on dataset side
        is_random_next = None

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # fp16 compatibility
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0

        text_embedding = self.word_embedding(input_ids, input_type_ids)
        text_embedding_total = self.process_text_embedding(
            text_embedding, input_mask == 0
        )

        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        # image_embedding_total = image_embedding_total *
        # input_mask.unsqueeze(-1).float()
        # text_embedding_total = text_embedding_total *
        # input_mask.unsqueeze(-1).float()

        if self.config.combine_embeddings:
            joint_embedding = self.combine_embeddings(
                ["image", "text"], [image_embedding_total, text_embedding_total]
            )
        else:
            joint_embedding = image_embedding_total

        output_dict = {}

        pooled_output = self.pooler(joint_embedding)

        if "pretraining" in self.config.training_head_type:
            prediction_scores, seq_relationship_score = self.classifier(
                joint_embedding, pooled_output
            )
            output_dict["logits"] = prediction_scores

            if masked_lm_labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(
                    prediction_scores.contiguous().view(
                        -1, self.bert_config.vocab_size
                    ),
                    masked_lm_labels.contiguous().view(-1),
                )
                # print(seq_relationship_score.argmax(dim=1), is_random_next)
                loss_key = "{}/{}".format(
                    sample_list.dataset_name, sample_list.dataset_type
                )

                output_dict["losses"] = {}
                output_dict["losses"][loss_key + "/masked_lm_loss"] = masked_lm_loss

                if is_random_next is not None:
                    output_dict["seq_relationship_score"] = seq_relationship_score

                    next_sentence_loss = loss_fct(
                        seq_relationship_score.contiguous().view(-1, 2),
                        is_random_next.contiguous().view(-1),
                    )
                    output_dict["losses"][
                        loss_key + "/next_sentence_loss"
                    ] = next_sentence_loss
            return output_dict
        elif (
            "vqa" in self.config.training_head_type
            or self.config.training_head_type == "vizwiz"
        ):
            index_to_gather = input_mask.sum(1) - 2

            pooled_output = torch.gather(
                joint_embedding,
                1,
                index_to_gather.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(index_to_gather.size(0), 1, joint_embedding.size(-1)),
            )

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            reshaped_logits = logits.contiguous().view(-1, self.answer_space_size)

            output_dict["scores"] = reshaped_logits
            return output_dict
        elif (
            self.config.training_head_type == "nlvr2"
            or self.config.training_head_type == "visual_entailment"
        ):
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            output_dict["scores"] = logits
            return output_dict

        return output_dict
