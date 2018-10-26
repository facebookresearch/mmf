import torch

from pythia.models.top_down_bottom_up import VQAMultiModalModel
from pythia.core.registry import Registry


@Registry.register_model('vizwiz_top_down_bottom_up')
class VizWizMultiModalModel(VQAMultiModalModel):
    def __init__(self, config):
        super(VizWizMultiModalModel, self).__init__(config)

    def build(self):
        self._init_text_embedding()
        # Initialize fasttext embedding for context
        self._init_context_embedding()

        # Initialize embeddings that take in feature embeddings from fasttext
        # for context
        self._init_feature_encoders("image")
        self._init_feature_encoders("context")
        self._init_feature_embeddings("context")
        self._init_feature_embeddings("image")
        self._init_combine_layer("image", "text")
        self._init_combine_layer("context", "text")
        self._init_classifier(self._get_classifier_input_dim())
        self._init_extras()

    def _init_context_embedding(self):
        self._init_text_embedding("context_embeddings")

    def _get_classifier_input_dim(self):
        return self.image_text_multi_modal_combine_layer.out_dim \
                + self.context_text_multi_modal_combine_layer.out_dim

    def get_optimizer_parameters(self, config):
        params = super(VizWizMultiModalModel, self).get_optimizer_parameters(
                    config
                 )

        combine_layer = self.context_text_multi_modal_combine_layer
        params.append({'params': combine_layer.parameters()})
        return params

    def forward(self, image_features, texts, contexts, info={}, **kwargs):
        input_text_variable = texts
        image_dim_variable = info.get('image_dim', None)
        image_feature_variables = image_features
        text_embedding_total = self.process_text_embedding(input_text_variable)

        context_embeddings = self.process_text_embedding(contexts,
                                                         'context_embeddings')
        context_dim_variable = info.get('context_dim', None)

        assert (len(image_feature_variables) ==
                len(self.image_feature_encoders)), \
            "number of image feature model doesnot equal \
             to number of image features"

        image_embedding_total = self.process_feature_embedding(
            "image",
            image_feature_variables,
            image_dim_variable,
            text_embedding_total
        )

        context_embedding_total = self.process_feature_embedding(
            "context",
            context_embeddings,
            context_dim_variable,
            text_embedding_total
         )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        image_text_embedding = self.combine_embeddings(
            ["image", "text"],
            [image_embedding_total, text_embedding_total]
        )

        context_text_embedding = self.combine_embeddings(
            ["context", "text"],
            [context_embedding_total, text_embedding_total]
        )

        joint_embedding = torch.cat([image_text_embedding,
                                     context_text_embedding], dim=1)

        return self.calculate_logits(joint_embedding)
