# Copyright (c) Facebook, Inc. and its affiliates.

# MMBTModel, ModalEmbeddings is copied from [1]
# as we have internal dependency on transformers v2.3.
# These will be removed when we upgrade to package v2.5+.
# [1]: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_mmbt.py # noqa

import torch
from mmf.common.registry import registry
from mmf.models.mmbt import MMBT, MMBTBase, MMBTConfig, MMBTModel, MMBTForClassification
from mmf.models.interfaces.retriever import RetrieverInterface


# TODO: Remove after transformers package upgrade to 2.5
class MMBTModelRAG(MMBTModel, RetrieverInterface):

    def __init__(self, config, transformer, encoder):
        super().__init__(config, transformer, encoder)
        self.config = config

    def proj_modal_embeddings(self, input_modal):
        modal_embeds = self.modal_encoder.encoder(input_modal)
        proj_embeds = self.modal_encoder.proj_embeddings(modal_embeds)
        return proj_embeds

    def ref_encode_image(self, input_modal):
        proj_embeds = self.proj_modal_embeddings(input_modal)
        return proj_embeds

    def ref_encode_text(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        token_type_ids=None,
        encoder_hidden_states=None,
    ):

        if input_ids is not None:
            input_txt_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_txt_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is None:
            token_type_ids = torch.ones(
                input_txt_shape, dtype=torch.long, device=input_ids.device
            )

        txt_embeddings = self.transformer.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        masks = self.construct_masks(
            txt_embeddings,
            input_modal_shape=None,
            attention_mask=None,
            encoder_attention_mask=None,
            head_mask=None
        )

        attention_mask, encoder_attention_mask, head_mask = masks

        encoder_outputs = self.transformer.encoder(
            txt_embeddings,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        # [bs,131,768]
        sequence_output = encoder_outputs[0]
        # [bs,768]
        pooled_output = self.transformer.pooler(sequence_output)

        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:
        ]

        return outputs[1]


class MMBTBaseRAG(MMBTBase, RetrieverInterface):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def build(self):
        super().build()
        encoders = self._build_encoders(self.config)
        text_encoder, modal_encoder = encoders[0], encoders[1]
        self._encoder_config = text_encoder.config

        self._mmbt_config = MMBTConfig(
            self._encoder_config,
            num_labels=self.config.num_labels,
            modal_hidden_size=self.config.modal_hidden_size,
        )

        self.mmbt = MMBTModelRAG(self._mmbt_config, text_encoder, modal_encoder)

    def ref_encode_image(self, input_modal):
        return self.mmbt.ref_encode_image(input_modal)

    def ref_encode_text(self, input_ids, segment_ids):
        return self.mmbt.ref_encode_text(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            position_ids=None,
            inputs_embeds=None,
        )


class MMBTForClassificationRAG(MMBTForClassification, RetrieverInterface):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.bert = MMBTBaseRAG(config, *args, **kwargs)

    def ref_encode_image(self, input_modal):
        return self.bert.ref_encode_image(input_modal)

    def ref_encode_text(self, input_ids, segment_ids):
        return self.bert.ref_encode_text(input_ids, segment_ids)

    def ref_encode_image_text(self, batch):
        # same as line pooled_output = module_output[1] in foward
        return self.bert(batch)[1]


@registry.register_model("mmbt_rag")
class MMBTRAG(MMBT, RetrieverInterface):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        self.model = MMBTForClassificationRAG(self.config)

    def ref_encode_image(self, input_modal):
        return self.model.ref_encode_image(input_modal)

    def ref_encode_text(self, input_ids, segment_ids):
        return self.model.ref_encode_text(input_ids, segment_ids)
