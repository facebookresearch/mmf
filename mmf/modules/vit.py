# Copyright (c) Facebook, Inc. and its affiliates.

from collections import namedtuple
from dataclasses import asdict, dataclass

import torch
from mmf.utils.general import retry_n
from omegaconf import OmegaConf
from packaging import version
from torch import nn

try:
    from transformers3 import __version__ as transformers_version
    from transformers3.modeling_bert import BertSelfAttention
except ImportError:
    from transformers import __version__ as transformers_version
    from transformers.modeling_bert import BertSelfAttention


if version.parse(transformers_version) >= version.parse("4.5.0"):
    try:
        import transformers3.models.vit.modeling_vit as vit
    except ImportError:
        import transformers.models.vit.modeling_vit as vit

    has_VIT = True
else:
    ViTStub = namedtuple("Vit", ["ViTAttention", "ViTPreTrainedModel"])
    vit = ViTStub(torch.nn.Module, torch.nn.Module)
    has_VIT = False


def check_vit_in_transformers():
    if not has_VIT:
        raise ImportError(
            "transformers version >= 4.5.0 required for using modeling_vit"
        )


NUM_RETRIES = 6


class ViTAttention(vit.ViTAttention):
    def __init__(self, config):
        check_vit_in_transformers()
        super().__init__(config)
        # We need to support attention masks for vision language input
        # ViTAttention from transformers doesn't currently support attention masks,
        # for versions without attention_mask support we use these clones of ViT modules
        # that use BertSelfAttention to enable masking.
        self.attention = BertSelfAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = vit.ViTIntermediate(config)
        self.output = vit.ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            self.layernorm_before(
                hidden_states
            ),  # in ViT, layernorm is applied before self-attention
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        outputs = (layer_output,) + outputs

        return outputs


class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [ViTLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    use_reentrant=True,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, layer_head_mask, output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return vit.BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTModel(vit.ViTPreTrainedModel):
    @dataclass
    class Config:
        name: str = "vit"
        # See https://huggingface.co/models?filter=vit for available options
        pretrained_model_name: str = "google/vit-base-patch16-224"
        random_init: bool = False
        gradient_checkpointing: bool = False
        do_patch_embeddings: bool = True

    def __init__(self, config):
        check_vit_in_transformers()
        super().__init__(config)
        self.config = config

        self.embeddings = vit.ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        add_pooling_layer = getattr(config, "add_pooling_layer", True)
        self.pooler = vit.ViTPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads
        to prune in this layer} See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_values=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import ViTFeatureExtractor, ViTModel
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = ViTFeatureExtractor.from_pretrained(
                    'google/vit-base-patch16-224-in21k'
                )
            >>> model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_values is None:
            raise ValueError("You have to specify input_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        do_patch_embeddings = getattr(self.config, "do_patch_embeddings", True)
        embedding_output = (
            self.embeddings(input_values) if do_patch_embeddings else input_values
        )

        batch_size, seq_length, _ = embedding_output.shape
        device = embedding_output.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, (batch_size, seq_length), device
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return vit.BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @staticmethod
    def from_config(config: Config):
        check_vit_in_transformers()

        config_with_defaults = OmegaConf.create({**asdict(ViTModel.Config()), **config})
        random_init = config_with_defaults.get("random_init", False)

        hf_config = retry_n(
            NUM_RETRIES,
            vit.ViTConfig.from_pretrained,
            config_with_defaults.pretrained_model_name,
            **OmegaConf.to_container(config_with_defaults),
        )
        hf_config.update(config)

        if not random_init:
            module = retry_n(
                NUM_RETRIES,
                ViTModel.from_pretrained,
                config.pretrained_model_name,
                config=hf_config,
            )
        else:
            module = ViTModel(hf_config)

        return module, hf_config
