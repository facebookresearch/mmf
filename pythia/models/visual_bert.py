import math
import os
import torch

from copy import deepcopy
from torch import nn
from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertPooler,
    BertLayer, BertPreTrainedModel, BertPreTrainingHeads
)
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from pythia.common.registry import registry
from pythia.models import BaseModel


class BertEmbeddingsWithVisualEmbedding(BertEmbeddings):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.token_type_embeddings_visual = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings_visual = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    def special_intialize(self, method_type = 0):
        ### This is a bit unorthodox. The better way might be to add an inititilizer to AllenNLP.
        # This function is used to initialize the token_type_embeddings_visual and positiona_embedding_visual, just incase.
        self.token_type_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.token_type_embeddings.weight.data), requires_grad=True
        )
        self.position_embeddings_visual.weight = nn.Parameter(
            deepcopy(self.position_embeddings.weight.data), requires_grad=True
        )
        return

    def forward(self, input_ids, token_type_ids=None, visual_embeddings=None,
                visual_embeddings_type=None, position_embeddings_visual=None,
                image_text_alignment = None, confidence = None):
        '''
        input_ids = [batch_size, sequence_length]
        token_type_ids = [batch_size, sequence_length]
        visual_embedding = [batch_size, image_feature_length, image_feature_dim]
        image_text_alignment = [batch_size, image_feature_length, alignment_dim]
        confidence = [batch_size, image_feature_length] of type LongTensor
        '''

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if visual_embeddings is not None:
            visual_embeddings = self.projection(visual_embeddings)
            token_type_embeddings_visual = self.token_type_embeddings_visual(visual_embeddings_type)

            if image_text_alignment is not None:
                # image_text_alignment = Batch x image_length x alignment_number.
                # Each element denotes the position of the word corresponding to the image
                # feature. -1 is the padding value.
                image_text_alignment_mask = (image_text_alignment != -1).long()
                # Get rid of the -1.
                image_text_alignment = image_text_alignment_mask * image_text_alignment

                # position_embeddings_visual = Batch x image_length x alignment length x dim
                position_embeddings_visual = self.position_embeddings(image_text_alignment) \
                    * image_text_alignment_mask.to(dtype=next(self.parameters()).dtype).unsqueeze(-1)
                position_embeddings_visual = position_embeddings_visual.sum(2)

                # We want to averge along the alignment_number dimension.
                image_text_alignment_mask = image_text_alignment_mask.to(dtype=next(self.parameters()).dtype).sum(2)
                image_text_alignment_mask[image_text_alignment_mask==0] = 1 # Avoid devide by zero error
                position_embeddings_visual = position_embeddings_visual / image_text_alignment_mask.unsqueeze(-1)

                position_ids_visual = torch.zeros(*visual_embeddings.size()[:-1], dtype = torch.long).cuda()

                # When fine-tuning the detector , the image_text_alignment is sometimes padded too long.
                if position_embeddings_visual.size(1) != visual_embeddings.size(1):
                    assert(position_embeddings_visual.size(1) >= visual_embeddings.size(1))
                    position_embeddings_visual = position_embeddings_visual[:, :visual_embeddings.size(1), :]

                position_embeddings_visual = position_embeddings_visual + self.position_embeddings_visual(position_ids_visual)
            else:
                position_ids_visual = torch.zeros(*visual_embeddings.size()[:-1], dtype = torch.long).cuda()
                position_embeddings_visual = self.position_embeddings_visual(position_ids_visual)

            v_embeddings = visual_embeddings + position_embeddings_visual + token_type_embeddings_visual

            # Concate the two:
            embeddings = torch.cat((embeddings, v_embeddings), dim = 1) # concat the visual embeddings after the attentions

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertVisualModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertVisualModel, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddingsWithVisualEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = BertLayer(config)

        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.fixed_head_masks = [None for _ in range(len(self.encoder.layer))]
        self.apply(self.init_weights)

    def forward(
        self, input_ids, token_type_ids, attention_mask, visual_embeddings,
        position_embeddings_visual, visual_embeddings_type, image_text_alignment,
        confidence
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids, token_type_ids, visual_embeddings=visual_embeddings,
            position_embeddings_visual=position_embeddings_visual,
            visual_embeddings_type=visual_embeddings_type,
            image_text_alignment=image_text_alignment,
            confidence=confidence
        )

        if self.bypass_transformer and visual_embeddings is not None:
            assert(not self.output_hidden_states) # Don't support this for the bypass model
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_part = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[:, :, :text_length, :text_length]

            encoded_layers = self.encoder(
                text_embedding_output, text_extended_attention_mask, self.fixed_head_masks
            )
            sequence_output = encoded_layers[0]
            new_input = torch.cat((sequence_output, visual_part), dim = 1)
            final_sequence_output = self.additional_layer(new_input, extended_attention_mask)
            pooled_output = self.pooler(final_sequence_output)
            return final_sequence_output, pooled_output

        if self.output_attentions:
            encoded_layers = self.encoder(
                embedding_output, extended_attention_mask, self.fixed_head_masks
            )
            sequence_output = encoded_layers[0]
            attn_data_list = encoded_layers[1:]
            pooled_output = self.pooler(sequence_output)
            return encoded_layers, pooled_output, attn_data_list
        else:
            encoded_layers = self.encoder(
                embedding_output, extended_attention_mask, self.fixed_head_masks
            )
            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)
            return sequence_output, pooled_output

class TrainVisualBERTObjective(BertPreTrainedModel):
    def __init__(self, config, training_head_type, visual_embedding_dim = 512,
                 hard_cap_seq_len=None, cut_first="text", embedding_strategy="plain",
                 bypass_transformer=False, random_initialize=False,
                 output_attentions=False, output_hidden_states=False):
        super(TrainVisualBERTObjective, self).__init__(config)
        config.visual_embedding_dim = visual_embedding_dim

        config.embedding_strategy = embedding_strategy

        config.bypass_transformer = bypass_transformer

        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states

        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        self.cut_first = cut_first
        self.hard_cap_seq_len = hard_cap_seq_len
        self.bert = BertVisualModel(config)

        self.training_head_type = training_head_type
        if "pretraining" in self.training_head_type:
            self.cls = BertPreTrainingHeads(config)
        elif self.training_head_type == "multichoice":
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.num_choices = 4 # For VCR

        if "vqa" in self.training_head_type:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, 3129)
        elif self.training_head_type == "vqa_advanced":
            self.cls = BertPreTrainingHeads(config)
        elif self.training_head_type == "nlvr2":
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size * 2, 2)
        elif self.training_head_type == "flickr":
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.cls = BertPreTrainingHeads(config)
            self.flickr_attention = FlickrAttention(config)

        if random_initialize is False:
            self.apply(self.init_weights)
            self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        if hasattr(self, 'cls'):
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                    self.bert.embeddings.word_embeddings)

    def flatten(self, to_be_flattened={}, to_be_flattened_dim={}):
        flattened = {}
        for key in to_be_flattened.keys():
            flattened[key] = transform_to_batch_sequence(to_be_flattened[key])
        for key in to_be_flattened_dim.keys():
            flattened[key] = transform_to_batch_sequence_dim(to_be_flattened_dim[key])

        if flattened["visual_embeddings_type"] is None:
            if flattened["image_mask"] is not None:
                flattened["visual_embeddings_type"] = torch.zeros_like(
                    flattened["image_mask"], dtype = torch.long
                )
            else:
                flattened["visual_embeddings_type"] = None


        if flattened["image_mask"] is not None:
            attention_mask = torch.cat(
                (flattened["input_mask"], flattened["image_mask"]), dim = -1
            )

            assert(flattened["image_lm_labels"] is None) # Do not support this yet
            if flattened["masked_lm_labels"] is not None:
                assert(
                    flattened["masked_lm_labels"].size(-1) ==
                    flattened["input_mask"].size(-1)
                )
                new_lm_labels = torch.ones_like(attention_mask) * -1
                size_masked_lm_labels = flattened["masked_lm_labels"].size()
                assert(len(size_masked_lm_labels) == 2)
                new_lm_labels[
                    :size_masked_lm_labels[0], :size_masked_lm_labels[1]
                ] = flattened["masked_lm_labels"]
                flattened["masked_lm_labels"] = new_lm_labels
        else:
            attention_mask = flattened["input_mask"]

        flattened["attention_mask"] = attention_mask

        return flattened

    def forward_heads(
        self,
        output,
        flattened
    ):
        output_dict = {}

        sequence_output, pooled_output = output[0], output[1]
        if self.output_attentions:
            output_dict = {}
            output_dict["attention_weights"] = output[2]
            output_dict["losses"] = None
            return output_dict

        if self.output_hidden_states:
            output_dict["sequence_output"] = sequence_output
            output_dict["pooled_output"] = pooled_output
            output_dict["losses"] = None
            return output_dict

        if "pretraining" in self.training_head_type and "masked" in dataset_name:
            prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

            if flattened["masked_lm_labels"] is not None and is_random_next is not None:
                output_dict["logits"] = prediction_scores
                output_dict["seq_relationship_score"] = seq_relationship_score
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(
                    prediction_scores.contiguous().view(-1, self.config.vocab_size),
                    flattened["masked_lm_labels"].contiguous().view(-1)
                )
                next_sentence_loss = loss_fct(
                    seq_relationship_score.contiguous().view(-1, 2),
                    is_random_next.contiguous().view(-1)
                )
                output_dict["next_sentence_loss"] = next_sentence_loss
                output_dict["masked_lm_loss"] = masked_lm_loss
                output_dict["loss"] = masked_lm_loss + next_sentence_loss

            if flattened["masked_lm_labels"] is not None and is_random_next is None:
                output_dict["logits"] = prediction_scores
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(
                    prediction_scores.contiguous().view(-1, self.config.vocab_size),
                    flattened["masked_lm_labels"].contiguous().view(-1)
                )
                #output_dict["next_sentence_loss"] = None
                output_dict["masked_lm_loss"] = masked_lm_loss
                output_dict["loss"] = masked_lm_loss

            return output_dict

        elif self.training_head_type == "multichoice":
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            reshaped_logits = logits.contiguous().view(-1, self.num_choices)

            output_dict["logits"] = reshaped_logits
            output_dict["loss"] = None

            if label is not None:
                loss_fct = nn.CrossEntropyLoss()
                output_dict["loss"] = loss_fct(reshaped_logits, label.contiguous())

            return output_dict

        elif "vqa" in self.training_head_type:
            index_to_gather = flattened["input_mask"].sum(1) - 2

            pooled_output = torch.gather(
                sequence_output, 1,
                index_to_gather.unsqueeze(-1).unsqueeze(-1).expand(
                    index_to_gather.size(0), 1, sequence_output.size(-1)
                )
            )

            flattened["input_ids"] = torch.gather(
                flattened["input_ids"], 1,
                index_to_gather.unsqueeze(-1).expand(
                    index_to_gather.size(0), 1
                )
            )

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            reshaped_logits = logits.contiguous().view(-1, 3129)

            output_dict["scores"] = reshaped_logits
            # output_dict["loss"] = None
            # output_dict["accuracy"] = None

            # if label is not None:
            #     loss_fct = torch.nn.KLDivLoss(reduction = "batchmean")
            #     log_softmax = torch.nn.LogSoftmax(dim=-1)
            #     reshaped_logits = log_softmax(reshaped_logits)
            #     output_dict["loss"] = loss_fct(reshaped_logits, label.contiguous())

            #     output_dict["accuracy"] = torch.sum(compute_score_with_logits(reshaped_logits, label)) / label.size(0)

            return output_dict

        # elif self.training_head_type == "vqa_advanced":
        #     prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        #     output_dict["logits"] = prediction_scores
        #     output_dict["seq_relationship_score"] = seq_relationship_score
        #     output_dict["loss"] = None

        #     loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        #     masked_lm_loss = loss_fct(
        #         prediction_scores.contiguous().view(-1, self.config.vocab_size),
        #         flattened["masked_lm_labels"].contiguous().view(-1)
        #     )
        #     output_dict["masked_lm_loss"] = masked_lm_loss
        #     output_dict["loss"] = masked_lm_loss

        #     prediction_tokens = torch.max(prediction_scores, -1)[1].view(
        #         input_ids.size(0), -1
        #     ).cpu().numpy() # batch x sequence length , records the predicted words
        #     lm_labels = flattened["masked_lm_labels"].view(input_ids.size(0), -1).cpu().numpy()
        #     counter = 0.0
        #     flags = []
        #     for i in range(lm_labels.shape[0]):
        #         flag = True
        #         for j in range(lm_labels.shape[1]):
        #             if lm_labels[i][j] != -1 and prediction_tokens[i][j] != lm_labels[i][j]:
        #                 flag = False
        #                 break
        #         if flag:
        #             counter += 1
        #         flags.append(flag)

        #     output_dict["accuracy"] = counter / prediction_tokens.shape[0]

        #     return output_dict

        elif self.training_head_type == "nlvr2":
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            reshaped_logits = logits.contiguous()

            output_dict["scores"] = logits
            return output_dict

        elif self.training_head_type == "flickr":
            if flickr_position is not None:
                entities_num = (flickr_position != -1).long().view(-1).sum(-1)
                flickr_position_mask = (flickr_position != -1).long()

                # Make the -1 become 0
                flickr_position = flickr_position * flickr_position_mask

                # Selected_positions = batch x selected position x dim
                selected_positions = batched_index_select(sequence_output, 1, flickr_position)

                # Visual Features = batch x visual_feature_length x dim
                visual_features = sequence_output[:, flattened["input_mask"].size(1): ,:]
                assert(visual_features.size(1) == flattened["image_mask"].size(1))

                scores = self.flickr_attention(
                    selected_positions, visual_features, flattened["image_mask"]
                )

                # scores = batch x selected position x visual_feature
                # scores = selected_positions.bmm(visual_features.transpose(1,2))
                loss_fct = torch.nn.KLDivLoss(reduction = "batchmean")
                log_softmax = torch.nn.LogSoftmax(dim=-1)
                scores = log_softmax(scores)

                label = label.contiguous()
                # label = batch x selected_postion x needed position
                output_dict["loss"] = loss_fct(scores, label)
                acc, upper_acc = compute_score_with_logits_flickr(scores, label)
                output_dict["accuracy"] = acc / entities_num
                output_dict["upperbound_accuracy"] = upper_acc / entities_num
                output_dict["entity_num"] = entities_num
            return output_dict

    def flatten_for_bert(
        self,
        input_ids,
        token_type_ids,
        input_mask,

        visual_embeddings,
        position_embeddings_visual,
        image_mask,
        image_text_alignment = None,
        confidence = None,

        visual_embeddings_type=None,
        label=None,
        flickr_position = None,
        masked_lm_labels=None,
        image_lm_labels=None,
        is_random_next=None,
        dataset_name=None,
        **kwargs
    ):
        to_be_flattened = {}
        to_be_flattened_dim = {}
        to_be_flattened["input_ids"] = input_ids
        to_be_flattened["token_type_ids"] = token_type_ids
        to_be_flattened["input_mask"] = input_mask
        to_be_flattened["image_mask"] = image_mask
        to_be_flattened["masked_lm_labels"] = masked_lm_labels
        to_be_flattened["image_lm_labels"] = image_lm_labels
        to_be_flattened["position_embeddings_visual"] = position_embeddings_visual
        to_be_flattened["confidence"] = confidence
        to_be_flattened_dim["image_text_alignment"] = image_text_alignment
        to_be_flattened_dim["visual_embeddings"] = visual_embeddings
        to_be_flattened["visual_embeddings_type"] = visual_embeddings_type

        # We want to convert everything into: batch x sequence_length x (dim).
        flattened = self.flatten(to_be_flattened, to_be_flattened_dim)
        return flattened

    def forward_bert(self, flattened):
        return self.bert(
            flattened["input_ids"],
            flattened["token_type_ids"],
            flattened["attention_mask"],
            visual_embeddings=flattened["visual_embeddings"],
            position_embeddings_visual=flattened["position_embeddings_visual"],
            visual_embeddings_type=flattened["visual_embeddings_type"],
            image_text_alignment=flattened["image_text_alignment"],
            confidence=flattened["confidence"]
        )

    def forward(
        self, *args, **kwargs
    ):
        flattened = self.flatten_for_bert(*args, **kwargs)
        output = self.forward_bert(flattened)

        if self.training_head_type == "nlvr2":
            output = list(output)
            pooled_output = output[1]
            # 2B * H => B * 2H
            b, h = pooled_output.size()
            pooled_output = torch.cat([pooled_output[:b // 2], pooled_output[b // 2:]], dim=1)
            output[1] = pooled_output

        return self.forward_heads(output, flattened)


@registry.register_model("visual_bert")
class VisualBERTFixedImageEmbedding(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.text_only = config.text_only
        self.training_head_type = config.training_head_type

    def build(self):
        self.bert = TrainVisualBERTObjective.from_pretrained(
                self.config.bert_model_name,
                cache_dir=os.path.join(
                    str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed_{}".format(-1)
                ),
                training_head_type=self.config.training_head_type,
                visual_embedding_dim=self.config.visual_embedding_dim,
                hard_cap_seq_len=self.config.hard_cap_seq_len,
                cut_first=self.config.cut_first,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                random_initialize=self.config.random_initialize,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states
        )
        if self.config.special_visual_initialize:
            self.bert.bert.embeddings.special_intialize()

        if getattr(self.config, "freeze_base", False):
            for p in self.bert.bert.parameters():
                p.requires_grad = False
        # if self.training_head_type == "nlvr2" or self.training_head_type == "multichoice":
        #     self._accuracy = CategoricalAccuracy()
        # if "vqa" in self.training_head_type:
        #     self._accuracy = Average()
        # if self.training_head_type == "flickr":
        #     self._accuracy = Average()

    def forward(self, sample_list):
        params = self.get_image_and_text_features(sample_list)
        params["visual_embeddings_type"] = getattr(sample_list, "visual_embeddings_type", None)
        params["label"] = getattr(sample_list, "label", None)

        if params["label"] is None:
            label = getattr(sample_list, "targets", None)
        # For flickr we also need to provide the position
        params["flickr_position"] = getattr(sample_list, "flickr_position", None)
        # pretraining labels
        params["masked_lm_labels"] = getattr(sample_list, "lm_label_ids", None)
        # pretraining labels
        # is_random_next = getattr(sample_list, "is_correct", None)
        # TODO(aps): Fix on dataset side
        params["is_random_next"] = None
        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        if params["visual_embeddings"] is not None and params["image_dim"] is not None:
            image_mask = torch.arange(
                params["visual_embeddings"].size(-2)
            ).expand(*params["visual_embeddings"].size()[:-1]).cuda()
            if len(params["image_dim"].size()) < len(image_mask.size()):
                params["image_dim"] = params["image_dim"].unsqueeze(-1)
                assert(len(params["image_dim"].size()) == len(image_mask.size()))
            image_mask = image_mask < params["image_dim"]
            params["image_mask"] = image_mask.long()
        else:
            params["image_mask"] = None

        params["dataset_name"] = sample_list.dataset_name
        params["position_embeddings_visual"] = None

        output_dict = self.bert(**params)

        if "pretraining" in self.training_head_type and "masked" in dataset_name:
            loss_key = "{}/{}".format(sample_list.dataset_name, sample_list.dataset_type)
            output_dict["losses"] = {}
            output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop("masked_lm_loss")
            if is_random_next is not None:
                output_dict["losses"][loss_key + "/next_sentence_loss"] = output_dict.pop("next_sentence_loss")

        return output_dict

    def get_image_and_text_features(self, sample_list):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids

        if self.training_head_type == "nlvr2":
            bert_input_ids = torch.cat([bert_input_ids, bert_input_ids])
            bert_input_mask = torch.cat([bert_input_mask, bert_input_mask])
            bert_input_type_ids = torch.cat([bert_input_type_ids, bert_input_type_ids])

            # image input
            img0 = getattr(sample_list, "img0", {})
            image_info = getattr(img0, "image_info_0", {})
            image_dim_variable_0 = getattr(image_info, "max_features", None)
            image_feat_variable_0 = getattr(img0, "image_feature_0", None)

            img1 = getattr(sample_list, "img1", {})
            image_info = getattr(img1, "image_info_0", {})
            image_dim_variable_1 = getattr(image_info, "max_features", None)
            image_feat_variable_1 = getattr(img1, "image_feature_0", None)

            image_feat_variable = torch.cat([image_feat_variable_0, image_feat_variable_1])
            image_dim_variable = torch.cat([image_dim_variable_0, image_dim_variable_1])
        else:
            image_info = getattr(sample_list, "image_info_0", {})
            image_dim_variable = getattr(image_info, "max_features", None)
            image_feat_variable = getattr(sample_list, "image_feature_0", None)

        return {
            "input_ids": bert_input_ids,
            "input_mask": bert_input_mask,
            "token_type_ids": bert_input_type_ids,
            "image_dim": image_dim_variable,
            "visual_embeddings": image_feat_variable
        }

    def get_metrics(self, reset: bool = False):
        if self.training_head_type == "nlvr2" or self.training_head_type == "multichoice" or "vqa" in self.training_head_type or self.training_head_type == "flickr":
            return {"accuracy": self._accuracy.get_metric(reset)}
        return {"accuracy": 0.0}

    def get_optimizer_parameters(self, config):
        # Pretraining has same LR for all of the parts
        if self.training_head_type == "pretraining":
            return self.get_bert_configured_parameters(self)

        # For finetuning setup, we have classifier
        lr = config.optimizer_attributes.params.lr
        vb_config = getattr(config.model_attributes, "visual_bert", {})
        finetune_lr_multiplier = getattr(vb_config, "finetune_lr_multiplier", 1)
        # Finetune the bert pretrained part with finetune_lr_multiplier if it is set
        parameters = self.get_bert_configured_parameters(
            self.bert.bert, lr * finetune_lr_multiplier
        )
        # Classifier will be trained on the normal lr
        parameters += self.get_bert_configured_parameters(
            self.bert.classifier, lr
        )

        return parameters

    def get_bert_configured_parameters(self, module, lr=None):
        param_optimizer = list(module.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ], "weight_decay": 0.01
            },
            {"params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ], "weight_decay": 0.0
            }
        ]

        if lr is not None:
            for p in optimizer_grouped_parameters:
                p["lr"] = lr

        return optimizer_grouped_parameters


    @staticmethod
    def compute_score_with_logits(logits, labels):
        logits = masked_unk_softmax(logits, 1, 0)
        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*labels.size())
        one_hots = one_hots.cuda() if use_cuda else one_hots
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = (one_hots * labels)
        return scores


class FlickrAttention(nn.Module):
    def __init__(self, config):
        super(FlickrAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = 1#config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, attention_mask):
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        # We don't need value layers
        #mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        #value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        attention_scores = attention_scores.squeeze(1)
        return attention_scores


def transform_to_batch_sequence(tensor):
    if tensor is not None:
        if len(tensor.size()) == 2:
            return tensor
        else:
            assert(len(tensor.size()) == 3)
            return tensor.contiguous().view(-1, tensor.size(-1))
    else:
        return None


def transform_to_batch_sequence_dim(tensor):
    if tensor is not None:
        if len(tensor.size()) == 3:
            return tensor
        else:
            assert(len(tensor.size()) == 4)
            return tensor.contiguous().view(-1, tensor.size(-2), tensor.size(-1))
    else:
        return None


def masked_unk_softmax(x, dim, mask_idx):
    x1 = nn.functional.softmax(x, dim=dim)
    x1[:, mask_idx] = 0
    x1_sum = torch.sum(x1, dim=1, keepdim=True)
    y = x1 / x1_sum
    return y


def compute_score_with_logits(logits, labels):
    logits = masked_unk_softmax(logits, 1, 0)
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros_like(labels)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def compute_score_with_logits_flickr(logits, labels, recall = 1):
    # Manually changed the recall here when evaluating... A bit clumsy

    labels_mask = (labels != 0.0).float()
    upper_bound_labels = labels.sum(-1).view(-1).sum(-1)
    labels = torch.ones_like(labels) * labels_mask

    if recall != 1:
        # Evaluation model. We could slow down.
        # labels = batch x seq x target length

        logits = logits.topk(k=recall, dim = -1)[1].data.cpu().numpy()

        counter = 0.0
        labels = labels.data.cpu().numpy()
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                possibles = logits[i][j]
                current_label = labels[i][j][possibles]
                if current_label.sum(-1) != 0:
                    counter += 1
        counter = torch.Tensor([counter]).cuda()
        return counter, upper_bound_labels

    logits = torch.max(logits, -1)[1].data # argmax
    logits = logits.unsqueeze(-1)
    scores = torch.gather(input = labels, dim = 2, index = logits)
    scores = scores.view(-1).sum(-1)
    return scores, upper_bound_labels


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out
