import torch

from pythia.models.vizwiz_multi_modal import VizWizMultiModalModel
from pythia.core.registry import registry
from pythia.modules.layers import ClassifierLayer


@registry.register_model('vizwiz_top_down_bottom_up_soft_copy')
class VizWizMultiModalModelSoftCopy(VizWizMultiModalModel):
    def __init__(self, config):
        super(VizWizMultiModalModelSoftCopy, self).__init__(config)
        self.use_cuda = registry.get('use_cuda')
        self.use_order_vectors = True
        self.use_ocr_info = registry.get('use_ocr_info')

    def build(self):
        super(VizWizMultiModalModelSoftCopy, self).build()
        self._build_context_classifier()

    def _build_context_classifier(self):
        layer = ClassifierLayer(
            self.config['context_classifier']['type'],
            in_dim=self._get_classifier_input_dim(),
            out_dim=1,
            **self.config['context_classifier']['params']
        )

        self.context_classifier = torch.nn.Sequential(
            layer,
            torch.nn.Sigmoid()
        )

    def get_optimizer_parameters(self, config):
        params = super(VizWizMultiModalModelSoftCopy,
                       self).get_optimizer_parameters(
                    config
                 )

        params.append({'params': self.context_classifier.parameters()})

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

        image_embedding_total, image_attentions = \
            self.process_feature_embedding(
                "image",
                image_feature_variables,
                image_dim_variable,
                text_embedding_total
            )

        context_embedding_total, context_attentions = \
            self.process_feature_embedding(
                "context",
                context_embeddings,
                context_dim_variable,
                text_embedding_total,
                {'order_vectors': info.get('order_vectors')}
             )

        if self.inter_model is not None:
            image_embedding_total = self.inter_model(image_embedding_total)

        joint_embedding = self.combine_embeddings(
            ["image", "text"],
            [image_embedding_total, text_embedding_total,
             context_embedding_total]
        )

        assert not torch.isnan(context_attentions[0]).any()
        info = {
            'context_attentions': context_attentions,
            'image_attentions': image_attentions
        }

        final_logits = self.calculate_logits(joint_embedding)

        # context_sigma = self.context_classifier(joint_embedding)
        #
        # answer_space_weights = torch.nn.functional.log_softmax(final_logits,
        #                                                        dim=1)
        # answer_space_weights = torch.exp(answer_space_weights)
        #
        # answer_space_importance = torch.autograd.Variable(
        #     torch.ones(*context_sigma.size()), requires_grad=False
        # )
        #
        # if context_sigma.is_cuda:
        #     answer_space_importance = answer_space_importance.cuda()
        # answer_space_importance = answer_space_importance - context_sigma
        # answer_space_importance = answer_space_importance.expand_as(
        #     answer_space_weights
        # )
        #
        # context_sigma = context_sigma.expand_as(
        #     context_attentions[0]
        # )
        #
        # attention = torch.nn.functional.log_softmax(context_attentions[0],
        #                                             dim=1)
        # attention = torch.exp(attention)
        # final_logits = torch.cat([
        #     answer_space_weights * answer_space_importance,
        #     attention * context_sigma
        # ], dim=1)

        return final_logits, info
