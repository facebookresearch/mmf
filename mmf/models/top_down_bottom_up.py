# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ReLUWithWeightNormFC


# Note: Doesn't work currently. Needs to be migrated to new API
@registry.register_model("top_down_bottom_up")
class TopDownBottomUp(BaseModel):
    def __init__(self, image_attention_model, text_embedding_models, classifier):
        super().__init__()
        self.image_attention_model = image_attention_model
        self.text_embedding_models = text_embedding_models
        self.classifier = classifier
        text_lstm_dim = sum([q.text_out_dim for q in text_embedding_models])
        joint_embedding_out_dim = classifier.input_dim
        image_feat_dim = image_attention_model.image_feat_dim
        self.non_linear_text = ReLUWithWeightNormFC(
            text_lstm_dim, joint_embedding_out_dim
        )
        self.non_linear_image = ReLUWithWeightNormFC(
            image_feat_dim, joint_embedding_out_dim
        )

    @classmethod
    def config_path(self):
        return None

    def build(self):
        return

    def forward(
        self, image_feat_variable, input_text_variable, input_answers=None, **kwargs
    ):
        text_embeddings = []
        for q_model in self.text_embedding_models:
            q_embedding = q_model(input_text_variable)
            text_embeddings.append(q_embedding)
        text_embedding = torch.cat(text_embeddings, dim=1)

        if isinstance(image_feat_variable, list):
            image_embeddings = []
            for idx, image_feat in enumerate(image_feat_variable):
                ques_embedding_each = torch.unsqueeze(text_embedding[idx, :], 0)
                image_feat_each = torch.unsqueeze(image_feat, dim=0)
                attention_each = self.image_attention_model(
                    image_feat_each, ques_embedding_each
                )
                image_embedding_each = torch.sum(attention_each * image_feat, dim=1)
                image_embeddings.append(image_embedding_each)
            image_embedding = torch.cat(image_embeddings, dim=0)
        else:
            attention = self.image_attention_model(image_feat_variable, text_embedding)
            image_embedding = torch.sum(attention * image_feat_variable, dim=1)

        joint_embedding = self.non_linear_text(text_embedding) * self.non_linear_image(
            image_embedding
        )
        logit_res = self.classifier(joint_embedding)

        return logit_res
