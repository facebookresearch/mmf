# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import math
import os

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.embeddings import BertVisioLinguisticEmbeddings
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.file_io import PathManager
from mmf.utils.modeling import get_optimizer_parameters_for_bert
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from omegaconf import OmegaConf
from torch import nn

try:
    from transformers3.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertLayer,
        BertModel,
        BertPooler,
        BertPredictionHeadTransform,
        BertPreTrainedModel,
    )
except ImportError:
    from transformers.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertLayer,
        BertModel,
        BertPooler,
        BertPredictionHeadTransform,
        BertPreTrainedModel,
    )


logger = logging.getLogger(__name__)


# This model essentially wraps GraphNetworkModule and multi-modal models
@registry.register_model("krisp")
class KRISP(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.build()

    @classmethod
    def config_path(cls):
        return "configs/models/krisp/defaults.yaml"

    # Each method need to define a build method where the model's modules
    # are actually build and assigned to the model
    def build(self):
        # Get any cross-model info we need for building network
        # (like hidden sizes)
        extra_config = {}
        extra_config["vb_hid_sz"] = self.config.visual_bert.hidden_size
        extra_config["node_hid_dim"] = self.config.graph_module.node_hid_dim

        # Also pass arguments to know if it needs to feed in something
        extra_config["feed_vb_to_graph"] = self.config.feed_vb_to_graph
        extra_config["feed_q_to_graph"] = self.config.feed_q_to_graph
        extra_config["feed_mode"] = self.config.feed_mode
        extra_config["feed_graph_to_vb"] = self.config.feed_graph_to_vb
        extra_config["feed_special_node"] = self.config.feed_special_node
        extra_config["topk_ans_feed"] = self.config.topk_ans_feed
        extra_config["compress_crossmodel"] = self.config.compress_crossmodel
        extra_config["crossmodel_compress_dim"] = self.config.crossmodel_compress_dim
        extra_config["analysis_mode"] = self.config.analysis_mode
        extra_config["noback_vb"] = self.config.noback_vb_to_graph

        # If feed q, make the question module here
        if self.config.feed_q_to_graph:
            # We can just make it a BERT model really easily
            self.q_enc = BertModel.from_pretrained("bert-base-uncased")
            extra_config["q_hid_sz"] = self.q_enc.config.hidden_size

        # Import graph network module
        # Putting in try-catch to avoid adding dependencies to mmf
        try:
            from projects.krisp.graphnetwork_module import GraphNetworkModule
        except Exception:
            print(
                "Import error with KRISP dependencies. Fix dependencies if "
                + "you want to use KRISP"
            )
            raise
        # Builds the graph network module
        self.graph_module = GraphNetworkModule(self.config.graph_module, extra_config)

        # Make VisualBERT module (without the final hidden logit layer)
        self.vb_module = VisualBERTModule(self.config.visual_bert, extra_config)

        # Final hidden layer for the vb module
        self.vocab_fc = nn.Linear(
            self.vb_module.model.bert.config.hidden_size, self.config.num_labels
        )

        # There's whether to use the bilinear and then whether to add or concat features
        # These are not mutally exclusive really
        # If output combine is ptr net, make GraphPtr Net for combining outputs
        if self.config.graph_logit_mode == "mc4":
            # Bilinear network
            self.graph_ptr_net = GraphPtrNet(
                self.vb_module.model.bert.config.hidden_size,
                self.config.graph_module.node_hid_dim,
            )
        elif self.config.graph_logit_mode == "in_graph":
            # Logits is already computed
            pass
        elif self.config.graph_logit_mode == "logit_fc":
            # Compute logits from single hidden layer
            self.graph_logit_fc = nn.Linear(
                self.config.graph_module.node_hid_dim, self.config.num_labels
            )

        # Answer indices not in graph
        if self.config.output_combine == "add":
            self.missing_ans_inds = torch.LongTensor(self.config.num_labels).fill_(1)
            self.missing_ans_inds[
                self.graph_module.index_in_ans
            ] = 0  # Now any index stil set to 1 is missing from graph

    # Each model in MMF gets a dict called sample_list which contains
    # all of the necessary information returned from the image
    def forward(self, sample_list):
        # If we have different combine modes, may need to call in different order
        if self.config.feed_graph_to_vb:
            # Can't be both (would create circular dep)
            assert not self.config.feed_vb_to_graph

            # Check mode
            # Can be feed_graph_hid_to_vb, where we pass in some vector
            # rep of graph into vb or feed_top_node_to_vb which is similar,
            # but it feeds in k node hidden states
            assert self.config.feed_mode in [
                "feed_graph_hid_to_vb",
                "feed_top_node_to_vb",
            ]
            if self.config.feed_mode == "feed_graph_hid_to_vb":
                assert self.graph_module.gn.output_special_node
            else:
                raise Exception("Unknown feed mode %s" % self.config.feed_mode)

            # Forward through graph module
            graph_output = self.graph_module(sample_list)

            # Put graph_output into sample_list
            sample_list["graph_output"] = graph_output

            # Forward through vb module
            vb_hidden = self.vb_module(sample_list)

            # Get vocab logit preds
            vb_logits = self.vocab_fc(vb_hidden)

        else:
            # Check mode
            if self.config.feed_vb_to_graph:
                # Can be feed_vb_hid_to_graph where we feed final
                # vb state into graph as a node input
                # Or feed_vb_logit_to_graph where we feed vg_predicted logits into graph
                assert self.config.feed_mode in [
                    "feed_vb_hid_to_graph",
                    "feed_vb_logit_to_graph",
                ]

            # Forward through vb module
            vb_hidden = self.vb_module(sample_list)

            # Get vocab logit preds
            vb_logits = self.vocab_fc(vb_hidden)

            sample_list["vb_hidden"] = vb_hidden
            sample_list["vb_logits"] = vb_logits

            # If we feed seperate Q feats into graph
            if self.config.feed_q_to_graph:
                # Now sample_list has all the processed inputs for us
                attention_mask_q = (sample_list["input_ids"] != 0).float()
                q_enc_out = self.q_enc(
                    input_ids=sample_list["input_ids"],
                    attention_mask=attention_mask_q,
                    token_type_ids=sample_list["token_type_ids"],
                )
                sample_list["q_encoded"] = q_enc_out[1]  # Get pooled output

            # Forward through graph module
            graph_output = self.graph_module(sample_list)

        # Compute graph logits
        if self.config.graph_logit_mode == "mc4":
            # Use bilinear network
            if self.config.noback_vb_to_blinear:
                graph_logits = self.graph_ptr_net(vb_hidden.detach(), graph_output)
            else:
                graph_logits = self.graph_ptr_net(vb_hidden, graph_output)

        elif self.config.graph_logit_mode == "in_graph":
            # Logits is already computed
            graph_logits = graph_output
            assert self.config.graph_module.output_type == "graph_prediction"
        elif self.config.graph_logit_mode == "logit_fc":
            # Compute logits from single hidden layer
            graph_logits = self.graph_logit_fc(graph_output)

        # Now combine outputs
        if self.config.output_combine == "concat":
            # Output order should be alphabetical
            assert self.config.graph_module.output_order == "alpha"

            # Combine both logits
            logits = torch.cat([vb_logits, graph_logits], dim=1)
        elif self.config.output_combine == "add":
            # Output order should be ans
            assert self.config.graph_module.output_order == "ans"

            # Set invalid inds to zero here
            assert graph_logits.size(1) == vb_logits.size(1)
            graph_logits[:, self.missing_ans_inds] = 0
            logits = vb_logits + graph_logits

        # Do zerobias
        if self.config.zerobias:
            logits -= 6.58

        # For loss calculations (automatically done by MMF
        # as per the loss defined in the config),
        # we need to return a dict with "scores" key as logits
        output = {"scores": logits}

        # If we're in eval / analysis mode, add more to output
        if self.config.analysis_mode:
            output = self.graph_module.add_analysis_to_output(output)

        # MMF will automatically calculate loss
        return output


class GraphPtrNet(nn.Module):
    def __init__(self, hidden_size, graph_hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.graph_hidden_size = graph_hidden_size

        self.bl_w = nn.Linear(hidden_size, hidden_size)
        self.graph_w = nn.Linear(graph_hidden_size, hidden_size)

    def forward(self, bl_hidden, graph_hidden):
        # Compute Eq. 4 from Iterative Answer Prediction with
        # Pointer-Augmented Multimodal Transformers for TextVQA
        # bl_hidden is bs x hidden_size
        # graph_hidden is bs x graph_hidden_size

        # Compute BL half
        bl_hidden = self.bl_w(bl_hidden)
        assert bl_hidden.dim() == 2
        bl_hidden = bl_hidden.unsqueeze(1)

        # Compute graph hidden half
        # Assume we've already subsampled to only valid answer nodes
        graph_hidden = self.graph_w(graph_hidden)

        # Now we have bl_hidden as a bs x 1 x hid vec
        # graph_hidden as a bs x num_nodes x hid vec

        # Combine
        scores = torch.matmul(bl_hidden, graph_hidden.transpose(-1, -2))

        # Normalize
        scores = scores / math.sqrt(self.hidden_size)
        scores = scores.squeeze(1)

        # Scores is now a bs x #nodes matrix
        return scores


class VisualBERTBase(BertPreTrainedModel):
    def __init__(
        self,
        config,
        visual_embedding_dim=512,
        embedding_strategy="plain",
        bypass_transformer=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config

        config.visual_embedding_dim = visual_embedding_dim
        config.embedding_strategy = embedding_strategy
        config.bypass_transformer = bypass_transformer
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states

        self.embeddings = BertVisioLinguisticEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.bypass_transformer = config.bypass_transformer

        if self.bypass_transformer:
            self.additional_layer = BertLayer(config)

        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.fixed_head_masks = [None for _ in range(len(self.encoder.layer))]
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        graph_input=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
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

        assert position_embeddings_visual is None
        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
            image_text_alignment=image_text_alignment,
        )

        if self.bypass_transformer and visual_embeddings is not None:
            assert (
                not self.output_hidden_states
            )  # Don't support this for the bypass model
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_part = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[
                :, :, :text_length, :text_length
            ]

            encoded_layers = self.encoder(
                text_embedding_output,
                text_extended_attention_mask,
                self.fixed_head_masks,
            )
            sequence_output = encoded_layers[0]
            new_input = torch.cat((sequence_output, visual_part), dim=1)
            final_sequence_output = self.additional_layer(
                new_input, extended_attention_mask
            )
            pooled_output = self.pooler(final_sequence_output)
            return final_sequence_output, pooled_output

        else:
            # If it takes graph input(s)
            # Do forward here through its own embedding
            # Then concat to embedding_output
            # And concat onto the extended_attention_mask to inclode this too
            if graph_input is not None:
                # Concat onto embeddings
                embedding_output = torch.cat([embedding_output, graph_input], dim=1)
                graph_att_mask = torch.zeros(
                    graph_input.size(0), 1, 1, graph_input.size(1)
                ).to(extended_attention_mask.device)
                extended_attention_mask = torch.cat(
                    [extended_attention_mask, graph_att_mask], dim=3
                )

            encoded_layers = self.encoder(
                embedding_output, extended_attention_mask, self.fixed_head_masks
            )
            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)
            attn_data_list = []

            if self.output_attentions:
                attn_data_list = encoded_layers[1:]

            return sequence_output, pooled_output, attn_data_list


class VisualBERTForClassification(nn.Module):
    def __init__(self, config, extra_config):
        super().__init__()
        self.config = config
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.pooler_strategy = self.config.get("pooler_strategy", "default")

        # Graph input params
        self.feed_graph_to_vb = extra_config["feed_graph_to_vb"]
        self.graph_node_hid_dim = extra_config["node_hid_dim"]
        self.graph_feed_mode = extra_config["feed_mode"]
        self.graph_topk = extra_config["topk_ans_feed"]

        # If doing graph, make a graph embedding layer
        if self.feed_graph_to_vb:
            self.graph_embedding = nn.Sequential(
                nn.Linear(self.graph_node_hid_dim, config.hidden_size),
                nn.LayerNorm(config.hidden_size, eps=1e-12),
                nn.Dropout(config.hidden_dropout_prob),  # hidden_dropout_prb
            )

        # If bert_model_name is not specified, you will need to specify
        # all of the required parameters for BERTConfig and a pretrained
        # model won't be loaded
        self.bert_model_name = self.config.get("bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        if self.bert_model_name is None or self.bert_model_name == "nopretrain":
            self.bert = VisualBERTBase(
                self.bert_config,
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            self.bert = VisualBERTBase.from_pretrained(
                self.config.bert_model_name,
                config=self.bert_config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
                visual_embedding_dim=self.config.visual_embedding_dim,
                embedding_strategy=self.config.embedding_strategy,
                bypass_transformer=self.config.bypass_transformer,
                output_attentions=self.config.output_attentions,
                output_hidden_states=self.config.output_hidden_states,
            )

        self.training_head_type = self.config.training_head_type
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        if self.config.training_head_type == "nlvr2":
            self.bert.config.hidden_size *= 2
        self.classifier = nn.Sequential(BertPredictionHeadTransform(self.bert.config))

        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()

            # Classifier needs to be initialized always as it is task specific
            self.classifier.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids,
        input_mask,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        masked_lm_labels=None,
        graph_input=None,
    ):
        # If we have a graph input, do the embedding first
        if self.feed_graph_to_vb:
            # Sanity check sizes
            if self.graph_feed_mode == "feed_graph_hid_to_vb":
                assert (
                    graph_input.dim() == 2
                    and graph_input.size(0) == input_ids.size(0)
                    and graph_input.size(1) == self.graph_node_hid_dim
                )
                graph_input = graph_input.unsqueeze(1)  # Add extra dim
            elif self.graph_feed_mode == "feed_top_node_to_vb":
                assert (
                    graph_input.dim() == 3
                    and graph_input.size(0) == input_ids.size(0)
                    and graph_input.size(1) == self.graph_topk
                    and graph_input.size(1) == self.graph_node_hid_dim
                )
            # Do the graph embedding
            graph_input = self.graph_embedding(graph_input)

        sequence_output, pooled_output, attention_weights = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            position_embeddings_visual,
            visual_embeddings_type,
            image_text_alignment,
            graph_input,
        )

        if self.training_head_type == "nlvr2":
            # 2B * H => B * 2H
            b, h = pooled_output.size()
            pooled_output = torch.cat(
                [pooled_output[: b // 2], pooled_output[b // 2 :]], dim=1
            )

        output_dict = {}
        if self.output_attentions:
            output_dict["attention_weights"] = attention_weights

        if self.output_hidden_states:
            output_dict["sequence_output"] = sequence_output
            output_dict["pooled_output"] = pooled_output

        if self.pooler_strategy == "vqa":
            # In VQA2 pooling strategy, we use representation from second last token
            index_to_gather = input_mask.sum(1) - 2
            pooled_output = torch.gather(
                sequence_output,
                1,
                index_to_gather.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(index_to_gather.size(0), 1, sequence_output.size(-1)),
            )

        pooled_output = self.dropout(pooled_output)
        output = self.classifier(pooled_output).squeeze(1)
        return output


class VisualBERTModule(nn.Module):
    def __init__(self, config, extra_config=None):
        super().__init__()
        self.config = config
        if extra_config is None:
            self.extra_config = {}
        else:
            self.extra_config = extra_config
        self.build()

    def build(self):
        assert self.config.training_head_type != "pretraining"
        self.model = VisualBERTForClassification(self.config, self.extra_config)

        if self.config.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

        # Initialize from pretrained model
        if self.config.load_from_pretrained:
            # Load the raw checkpoint
            pretrained_file = self.config.pretrained_file
            with PathManager.open(pretrained_file, "rb") as f:
                ckpt = torch.load(f, map_location=lambda storage, loc: storage)
            model_ckpt = ckpt["model"]

            # Remove "model" in fron of keys
            model_ckpt_new = {}
            for key in model_ckpt:
                if "bert" not in key:
                    continue
                model_ckpt_new[key.split("model.")[1]] = model_ckpt[key]
            model_ckpt = model_ckpt_new

            # Load the checkpoint
            incompatible_keys = self.model.load_state_dict(model_ckpt, strict=False)

            # Print any missing / wrong keys for debug
            if len(incompatible_keys.missing_keys) != 0:
                logger.warning(
                    f"Missing keys {incompatible_keys.missing_keys} in the"
                    + " checkpoint.\n"
                    + "If this is not your checkpoint, please open up an "
                    + "issue on MMF GitHub. \n"
                    + f"Unexpected keys if any: {incompatible_keys.unexpected_keys}"
                )
            if len(incompatible_keys.unexpected_keys) != 0:
                logger.warning(
                    "Unexpected keys in state dict: "
                    + f"{incompatible_keys.unexpected_keys} \n"
                    + "This is usually not a problem with pretrained models, but "
                    + "if this is your own model, please double check. \n"
                    + "If you think this is an issue, please open up a "
                    + "bug at MMF GitHub."
                )

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

        # Graph input params
        self.feed_graph_to_vb = self.extra_config["feed_graph_to_vb"]
        self.graph_node_hid_dim = self.extra_config["node_hid_dim"]
        self.graph_feed_mode = self.extra_config["feed_mode"]

        # Not implemented for this model
        if self.feed_graph_to_vb and self.extra_config["compress_crossmodel"]:
            assert False

    def flatten(self, sample_list, to_be_flattened=None, to_be_flattened_dim=None):
        if to_be_flattened is None:
            to_be_flattened = {}
        if to_be_flattened_dim is None:
            to_be_flattened_dim = {}
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            sample_list[key] = getattr(sample_list, key, None)
            sample_list[key] = transform_to_batch_sequence(sample_list[key])
        for key in to_be_flattened_dim:
            sample_list[key] = getattr(sample_list, key, None)
            sample_list[key] = transform_to_batch_sequence_dim(sample_list[key])

        if sample_list.visual_embeddings_type is None:
            if sample_list.image_mask is not None:
                sample_list.visual_embeddings_type = torch.zeros_like(
                    sample_list.image_mask
                )

        if sample_list.image_mask is not None:
            attention_mask = torch.cat(
                (sample_list.input_mask, sample_list.image_mask), dim=-1
            )
            if sample_list.masked_lm_labels is not None:
                assert sample_list.masked_lm_labels.size(
                    -1
                ) == sample_list.input_mask.size(-1)
                new_lm_labels = torch.ones_like(attention_mask) * -1
                size_masked_lm_labels = sample_list.masked_lm_labels.size()
                assert len(size_masked_lm_labels) == 2
                new_lm_labels[
                    : size_masked_lm_labels[0], : size_masked_lm_labels[1]
                ] = sample_list.masked_lm_labels
                sample_list.masked_lm_labels = new_lm_labels
        else:
            attention_mask = sample_list.input_mask

        sample_list.attention_mask = attention_mask

        return sample_list

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def flatten_for_bert(self, sample_list, **kwargs):
        to_be_flattened = [
            "input_ids",
            "token_type_ids",
            "input_mask",
            "image_mask",
            "masked_lm_labels",
            # "position_embeddings_visual",
            # "visual_embeddings_type",
        ]
        to_be_flattened_dim = ["visual_embeddings"]  # "image_text_alignment",

        # We want to convert everything into: batch x sequence_length x (dim).
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def update_sample_list_based_on_head(self, sample_list):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids

        if self.config.training_head_type == "nlvr2":
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

            image_feat_variable = torch.cat(
                [image_feat_variable_0, image_feat_variable_1]
            )
            image_dim_variable = torch.cat([image_dim_variable_0, image_dim_variable_1])
        else:
            image_info = getattr(sample_list, "image_info_0", {})
            image_dim_variable = getattr(image_info, "max_features", None)
            image_feat_variable = getattr(sample_list, "image_feature_0", None)

        sample_list.visual_embeddings = image_feat_variable
        sample_list.image_dim = image_dim_variable
        sample_list.input_ids = bert_input_ids
        sample_list.input_mask = bert_input_mask
        sample_list.token_type_ids = bert_input_type_ids
        return sample_list

    def add_custom_params(self, sample_list):
        visual_embeddings = getattr(sample_list, "visual_embeddings", None)
        image_dim = getattr(sample_list, "image_dim", None)
        # pretraining labels
        sample_list.masked_lm_labels = getattr(sample_list, "lm_label_ids", None)
        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        if visual_embeddings is not None and image_dim is not None:
            image_mask = torch.arange(
                visual_embeddings.size(-2), device=visual_embeddings.device
            ).expand(*visual_embeddings.size()[:-1])
            if len(image_dim.size()) < len(image_mask.size()):
                image_dim = image_dim.unsqueeze(-1)
                assert len(image_dim.size()) == len(image_mask.size())
            image_mask = image_mask < image_dim
            sample_list.image_mask = image_mask.long()
        else:
            sample_list.image_mask = None

        sample_list.position_embeddings_visual = None
        sample_list.visual_embeddings_type = None
        sample_list.image_text_alignment = None
        return sample_list

    # Backward compatibility for code from original VisualBERT
    @classmethod
    def format_state_key(cls, key):
        return (
            key.replace("bert.bert", "model.bert")
            .replace("bert.cls", "model.cls")
            .replace("bert.classifier", "model.classifier")
        )

    def forward(self, sample_list):
        sample_list = self.update_sample_list_based_on_head(sample_list)
        sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_bert(sample_list)

        if self.feed_graph_to_vb:
            if self.graph_feed_mode == "feed_graph_hid_to_vb":
                assert "graph_special_node_out" in sample_list
                graph_input = sample_list["graph_special_node_out"]
            else:
                assert False
        else:
            graph_input = None

        output = self.model(
            sample_list.input_ids,
            sample_list.input_mask,
            sample_list.attention_mask,
            sample_list.token_type_ids,
            sample_list.visual_embeddings,
            sample_list.position_embeddings_visual,
            sample_list.visual_embeddings_type,
            sample_list.image_text_alignment,
            sample_list.masked_lm_labels,
            graph_input,
        )

        return output
