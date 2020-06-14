# Copyright 2019 project LXMERT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from transformers.modeling_bert import (
    ACT2FN,
    BertConfig,  # removed this from lxmert class def
    BertIntermediate,  # removed this from lxmert class def
    BertLayerNorm,
    # BertLMPredictionHead, keep custom class here bc we want to load custom weight
    BertOutput,  # removed custom lxmert class bc nothing was changed
    BertPredictionHeadTransform,
    BertPreTrainedModel,  # got rid of custom LXMERT class temporarily
)

from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.modeling import get_optimizer_parameters_for_bert


"""
 redfine BertEmbedding to explictly set padding idx
 the forward reimplemntation for BertEmbedding: does not support input embedding
 add visuzliztion opetion for bert attention
 implement dynamic attention if needed
 inputs that vilbert base model has that we do not
 image_location,
 co_attention_mask=None,
 task_ids=None,
 output_all_encoded_layers=False,
 output_all_attention_masks=False,
 task specific tokens? for vilbert base
 what is .init_weights? for vilbert base
 ADD TO LXMERT CONFIG TOO FOR PRETRAINING
 task_mask_lm=config.task_mask_lm
 task_matched=congig.task_matched
 task_obj_predict=config_obj_predict
 visual_losses=config.visual_losses
 task_qa=config.task_qa
 self.num_answers = config.num_answers
 include mode in config
 also need to implemented VisualConfig from Hao's repo into general config and then
 subsequently modify sub LXMERT classes
 changed all LXRT class names to LXMERT just for the sake of ubiquity
 DONT KNOW WHAT THIS WOULD DO
 if self.task_specific_tokens:
    # extend the mask
    mask_tokens = input_txt.new().resize_(input_txt.size(0), 1).fill_(1)
    attention_mask = torch.cat([mask_tokens, attention_mask], dim=1)
 for pretrianing
 should we define the cross entropy class funciton in the init?
 changed num answers to num labels
 KEEP THE FOLLOWING TO MAKE NEW CONFIG FILE
 LXMERTBERTCONFIG
     def __init__(
         self,
         vocab_size_or_config_json_file,
         hidden_size=768,
         num_hidden_layers=12,
         num_attention_heads=12,
         intermediate_size=3072,
         hidden_act="gelu",
         hidden_dropout_prob=0.1,
         attention_probs_dropout_prob=0.1,
         max_position_embeddings=512,
         type_vocab_size=2,
         initializer_range=0.02,
         make sure we have config option for padding idx
         config.pad_token_id
         config.layer_norm_eps
 what is fusion method?
 class VisualConfig(object):
     VISUAL_LOSSES = ["obj", "attr", "feat"]
     def __init__(self, l_layers=12, x_layers=5, r_layers=0):
         self.l_layers = l_layers
         self.x_layers = x_layers
         self.r_layers = r_layers
         self.visual_feat_dim = 2048
         self.visual_pos_dim = 4
         self.obj_id_num = 1600
         self.attr_id_num = 400
         self.visual_losses = self.VISUAL_LOSSES
         self.visual_loss_config = {
             "obj": (self.obj_id_num, "ce", (-1,), 1 / 0.15),
             "attr": (self.attr_id_num, "ce", (-1,), 1 / 0.15),
             "feat": (2048, "l2", (-1, 2048), 1 / 0.15),
         }
     def set_visual_dims(self, feat_dim, pos_dim):
         self.visual_feat_dim = feat_dim
         self.visual_pos_dim = pos_dim
 VISUAL_CONFIG = VisualConfig()
 """


def gelu(x):

    """
    Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
    This version is used in original LXMERT Implementation,
    we redefine since gelu implementaions differ by pytorch version
    """

    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


ACT2FN["gelu"] = gelu


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size,
            padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size,
            padding_idx=0
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size,
            padding_idx=0 #I guess it is 0
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is
        # (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself,
        # thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertClassificationHead(nn.Module):
    def __init__(self, num_labels, hid_dim, training_head_type):
        super().__init__()

        if training_head_type == "nlvr2":
            in_dim = hid_dim * 2
            out_dim = 2
        else:
            in_dim = hid_dim
            out_dim = num_labels

        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, out_dim)
        )

    def forward(self, x):
        logit = self.logit_fc(x)
        return logit


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertVisualAnswerHead(nn.Module):
    def __init__(self, config, num_answers):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class BertVisualObjHead(nn.Module):
    def __init__(self, config, visual_losses):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # Decide the use of visual losses
        visual_losses = visual_losses.split(",")
        for loss in visual_losses:
            assert loss in VISUAL_CONFIG.VISUAL_LOSSES
        self.visual_losses = visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict(
            {
                key: nn.Linear(
                    config.hidden_size, VISUAL_CONFIG.visual_loss_config[key][0]
                )
                for key in self.visual_losses
            }
        )

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class LXMERTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(
        self, lang_input, lang_attention_mask, visn_input, visn_attention_mask
    ):
        # Cross Attention
        lang_att_output = self.visual_attention(
            lang_input, visn_input, ctx_att_mask=visn_attention_mask
        )
        visn_att_output = self.visual_attention(
            visn_input, lang_input, ctx_att_mask=lang_attention_mask
        )
        return lang_att_output, visn_att_output

    def self_att(
        self, lang_input, lang_attention_mask, visn_input, visn_attention_mask
    ):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(
            lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask
        )
        lang_att_output, visn_att_output = self.self_att(
            lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask
        )
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output


class VisualFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = config.visual_feat_dim
        pos_dim = config.visual_pos_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats, boxes = visn_input

        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        output = (x + y) / 2

        output = self.dropout(output)
        return output


class LXMERTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = VisualFeatEncoder(config)

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers
        print(
            "LXMERT encoder with %d l_layers, %d x_layers, and %d r_layers."
            % (self.num_l_layers, self.num_x_layers, self.num_r_layers)
        )

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        self.x_layers = nn.ModuleList(
            [LXMERTXLayer(config) for _ in range(self.num_x_layers)]
        )
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        )

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask=None
    ):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        visn_feats = self.visn_fc(visn_feats)

        # Run language layers
        for layer_module in self.layer:
            lang_feats = layer_module(lang_feats, lang_attention_mask)

        # Run relational layers
        for layer_module in self.r_layers:
            visn_feats = layer_module(visn_feats, visn_attention_mask)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            lang_feats, visn_feats = layer_module(
                lang_feats, lang_attention_mask, visn_feats, visn_attention_mask
            )

        return lang_feats, visn_feats


class LXMERTBase(BertPreTrainedModel):
    """LXMERT Model."""
    # added visual feature mask

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = LXMERTEncoder(config)
        self.pooler = BertPooler(config)
#         self.apply(self.init_bert_weights)
#         self.task_specific_tokens = config.task_specific_tokens

#         self.init_weights()
        # task specific tokens?
        # what is .init_weights?

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        visual_feats=None,
        visual_attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # DONT KNOW WHAT THIS WOULD DO
        # if self.task_specific_tokens:
        #    # extend the mask
        #    mask_tokens = input_txt.new().resize_(input_txt.size(0), 1).fill_(1)
        #    attention_mask = torch.cat([mask_tokens, attention_mask], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the
        # triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(
                1
            ).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            extended_visual_attention_mask = (
                1.0 - extended_visual_attention_mask
            ) * -10000.0
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Run LXMERT backbone
        lang_feats, visn_feats = self.encoder(
            embedding_output,
            extended_attention_mask,
            visn_feats=visual_feats,
            visn_attention_mask=extended_visual_attention_mask,
        )
        pooled_output = self.pooler(lang_feats)

        return (lang_feats, visn_feats), pooled_output


class LXMERTForPretraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Configuration
        self.config = config
        self.num_labels = config.num_labels
        # LXMERT backbone
        self.bert = LXMERTBase.from_pretrained(
            self.config.bert_model_name,
            config=BertConfig.from_dict(
                OmegaConf.to_container(self.config, resolve=True)
            ),
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
        )
        # Use of pre-training tasks
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa
        self.visual_losses = config.visual_losses

        # Pre-training heads
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )
        if self.task_obj_predict:
            self.obj_predict_head = BertVisualObjHead(config, self.visual_losses)
        if self.task_qa:
            self.answer_head = BertVisualAnswerHead(config, self.num_labels)

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.config.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)
            self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning
            them instead.
        """
        self._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        masked_lm_labels=None,
        visual_feats=None,
        pos=None,
        obj_labels=None,
        matched_label=None,
        ans=None,
        # they provide feature location
        # they provide image attention mask
        # image label and target
        # next sentence label
        # output all attention masks
    ):

        (lang_output, visn_output), pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, visual_feats=(visual_feats, pos),
        )

        lang_prediction_scores, cross_relationship_score = self.cls(
            lang_output, pooled_output
        )

        ## KEEP TRACK OF OUTPUTS HERE
        output = {}
        # if output_all_attention_masks:
        #     output["attention_weights"] = attention_weights

        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            answer_score = pooled_output[0][0]

        total_loss = 0.0
        loss_fct = CrossEntropyLoss(ignore_index=-1)

        if masked_lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = loss_fct(
                lang_prediction_scores.view(-1, lang_prediction_scores.size(-1)),
                masked_lm_labels.view(-1),
            )
            total_loss += masked_lm_loss
            output["masked_lm_loss"] = masked_lm_loss.detach()
        if matched_label is not None and self.task_matched:
            matched_loss = loss_fct(
                cross_relationship_score.view(-1, 2), matched_label.view(-1)
            )
            total_loss += matched_loss
            output["matched_loss"] = matched_loss.detach()
        if obj_labels is not None and self.task_obj_predict:
            loss_fcts = {
                "l2": SmoothL1Loss(reduction="none"),
                "ce": CrossEntropyLoss(ignore_index=-1, reduction="none"),
            }
            total_visn_loss = 0.0
            visn_prediction_scores_dict = self.obj_predict_head(visn_output)
            for key in config.visual_losses:
                label, mask_conf = obj_labels[key]
                (
                    output_dim,
                    loss_fct_name,
                    label_shape,
                    weight,
                ) = config.visual_loss_config[key]
                visn_loss_fct = loss_fcts[loss_fct_name]
                visn_prediction_scores = visn_prediction_scores_dict[key]
                visn_loss = visn_loss_fct(
                    visn_prediction_scores.view(-1, output_dim),
                    label.view(*label_shape),
                )
                if visn_loss.dim() > 1:  # Regression Losses
                    visn_loss = visn_loss.mean(1)

                visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
                total_visn_loss += visn_loss
                output["{}_loss".format(key)] = visn_loss.detach()

            total_loss += total_visn_loss
            output["total_visn_loss"] = total_visn_loss
        if ans is not None and self.task_qa:
            answer_loss = loss_fct(
                answer_score.view(-1, self.num_labels), ans.view(-1)
            )
            # Since this Github version pre-trains with QA loss from the beginning,
            # I exclude "*2" here to match the effect of QA losses.
            # Previous: (loss *0) for 6 epochs,
            # (loss *2) for 6 epochs.   (Used 10 instead of 6 in EMNLP paper)
            # Now     : (loss *1) for 12 epochs
            #
            # * 2       # Multiply by 2 because > half of the data will not have label
            total_loss += answer_loss
            output["answer_loss"] = answer_loss.detach()
        # add each loss to output
        # add answer score to output
        # add total loss to output
        output["anwer_score"] = answer_score.detach()
#         output["total_loss"] = total_loss
        return output


class LXMERTForClassification(nn.Module):
    def __init__(self, config, mode="lxr"):
        super().__init__()

        self.config = config
        self.mode = config.mode
        self.bert = LXMERTBase.from_pretrained(
            self.config.bert_model_name,
            config=BertConfig.from_dict(
                OmegaConf.to_container(self.config, resolve=True)
            ),
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
        )

        self.classifier = BertClassificationHead(
            config.num_labels,
            config.hidden_dim,
            config.training_head_type)

        self.init_weights()

    @property
    def dim(self):
        return 768

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.config.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()

            # Classifier needs to be initialized always as it is task specific
            self.classifier.apply(self.bert._init_weights)

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            visual_feats=None,
            visual_attention_mask=None):

        feat_seq, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            visual_feats=visual_feats,
            visual_attention_mask=visual_attention_mask,
        )

        # if output_all_attention_masks:
        #    output["attention_weights"] = attention_weights

        if self.training_head_type == "nlvr2":
            pooled_output = pooled_output.view(-1, pooled_output.size(1) * 2)

        logits = self.logit_fc(pooled_output)

        # i wonder if we need this?
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)

        return {"scores": reshaped_logits}

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join("%s_LXMERT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXMERT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module.") :]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)


@registry.register_model("lxmert")
class LXMERT(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
#         return "projects/lxmert/configs/vqa2/defaults.yaml"
        return "configs/models/lxmert/pretrain.yaml"

    # Backward compatibility
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("bert.bert", "model.bert")
            .replace("bert.cls", "model.cls")
            .replace("bert.classifier", "model.classifier")
        )

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = LXMERTForPretraining(self.config)
        else:
            self.model = LXMERTForClassification(self.config)

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def get_image_and_text_features(self, sample_list):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids
        masked_lm_labels = sample_list.lm_label_ids

        image_info = getattr(sample_list, "image_info_0", {})
        bbox = np.array(getattr(image_info, "bbox", None), dtype=np.float32)
        feats = getattr(sample_list, "image_feature_0", None)
        obj_label_variable = getattr(sample_list, "image_labels", None)


        img_h, img_w = img_info['img_h'], img_info['img_w']
        image_location = np.zeros(
            (bbox.shape[0], bbox.shape[1], 5), dtype=np.float32
        )
        image_location[:, :, :4] = bbox
        image_location[:, :, 4] = (
            (image_location[:, :, 3] - image_location[:, :, 1])
            * (image_location[:, :, 2] - image_location[:, :, 0])
            / (image_w * image_h)[:, None]
        )
        image_location[:, :, 0] = image_location[:, :, 0] / image_w[:, None]
        image_location[:, :, 1] = image_location[:, :, 1] / image_h[:, None]
        image_location[:, :, 2] = image_location[:, :, 2] / image_w[:, None]
        image_location[:, :, 3] = image_location[:, :, 3] / image_h[:, None]
        image_location_variable = torch.tensor(
            image_location, dtype=torch.float
        ).cuda()


        is_matched = 1
        if self.config.taskMatched:
            if random.random() < 0.5:
                is_matched = 0
                ssf = torch.randperm(bert_input_ids.size(0))
                bert_input_ids = bert_input_ids[ssf]
                bert_input_mask = bert_input_mask[ssf]
                bert_input_type_ids = bert_input_type_ids[ssf]
                masked_lm_labels = masked_lm_labels[ssf]

        return {
            "input_ids": bert_input_ids,
            "token_type_ids": bert_input_mask,
            "attention_mask": bert_input_type_ids,
            "masked_lm_labels": masked_lm_labels,
            "visual_feats": feats,
            "pos": image_location,                      
            "obj_labels": obj_label_variable,
            "matched_label": is_matched,    
            "ans": None
        }

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def forward(self, sample_list):
        params = self.get_image_and_text_features(sample_list)
        # pretraining labels
#         params["masked_lm_labels"] = getattr(sample_list, "lm_label_ids", None)
        # is_random_next = getattr(sample_list, "is_correct", None)
        # TODO(aps): Fix on dataset side
        # params["is_random_next"] = None

        # Prepare Mask
        if params["image_feature"] is not None and params["image_dim"] is not None:
            image_mask = (
                torch.arange(params["image_feature"].size(-2))
                .expand(*params["image_feature"].size()[:-1])
                .cuda()
            )
            if len(params["image_dim"].size()) < len(image_mask.size()):
                params["image_dim"] = params["image_dim"].unsqueeze(-1)
                assert len(params["image_dim"].size()) == len(image_mask.size())
            image_mask = image_mask < params["image_dim"]
            params["image_attention_mask"] = image_mask.long()
        else:
            params["image_attention_mask"] = None
        params.pop("image_dim")
       

        if self.config.training_head_type == "pretraining":

            output_dict = self.model(
                params["input_ids"], 
                params["token_type_ids"],
                params["attention_mask"], 
                params["masked_lm_labels"], 
                params["visual_feats"], 
                params["pos"],                      
                params["obj_labels"], 
                params["matched_label"],             
                params["ans"], 
            )
#             pass
            # WE WILL NEED TO FIX UP INPUT STUFF FOR THIS
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            output_dict["losses"] = {}
            
            output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
                "masked_lm_loss"
            )
            output_dict["losses"][loss_key + "/matched_loss"] = output_dict.pop(
                "matched_loss"
            )
            output_dict["losses"][loss_key + "/visn_loss"] = output_dict.pop(
                "visn_loss"
            )
            output_dict["losses"][loss_key + "/answer_loss"] = output_dict.pop(
                "answer_loss"
            )
        else:
            output_dict = self.model(
                params["input_ids"], #input_ids
                params["token_type_ids"], #token_type_ids
                params["attention_mask"], #attention_mask
                params["visual_feats"], #visual_feats
                params["pos"], #pos  
            )
        return output_dict