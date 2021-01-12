# Copyright (c) Facebook, Inc. and its affiliates.

import copy
from typing import Any, Dict, List, Optional, Tuple

import omegaconf
import torch
from mmf.common.registry import registry
from mmf.common.typings import DictConfig
from mmf.models.base_model import BaseModel
from mmf.modules.embeddings import (
    PreExtractedEmbedding,
    TextEmbedding,
    TwoBranchEmbedding,
)
from mmf.modules.layers import BranchCombineLayer, ClassifierLayer
from mmf.utils.build import build_image_encoder
from mmf.utils.general import filter_grads


@registry.register_model("movie_mcan")
class MoVieMcan(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/movie_mcan/defaults.yaml"

    def build(self):
        self.image_feature_dim = 2048
        self._build_word_embedding()
        self._init_text_embeddings("text")
        self._init_feature_encoders("image")
        self._init_feature_embeddings("image")
        self._init_combine_layer("image", "text")
        self._init_classifier(self._get_classifier_input_dim())
        self._init_extras()

    def _build_word_embedding(self):
        assert len(self._datasets) > 0
        text_processor = registry.get(self._datasets[0] + "_text_processor")
        vocab = text_processor.vocab
        self.word_embedding = vocab.get_embedding(torch.nn.Embedding, embedding_dim=300)

    def _init_text_embeddings(self, attr: str = "text"):
        if "embeddings" not in attr:
            attr += "_embeddings"

        module_config = self.config[attr]
        embedding_type = module_config.type
        embedding_kwargs = copy.deepcopy(module_config.params)
        self._update_text_embedding_args(embedding_kwargs)
        embedding = TextEmbedding(embedding_type, **embedding_kwargs)
        embeddings_out_dim = embedding.text_out_dim

        setattr(self, attr + "_out_dim", embeddings_out_dim)
        setattr(self, attr, embedding)

    def _update_text_embedding_args(self, args):
        # Add model_data_dir to kwargs
        args.model_data_dir = self.config.model_data_dir

    def _init_feature_encoders(self, attr: str):
        feat_encoder = self.config[attr + "_feature_encodings"]
        feature_dim = self.config[attr + "_feature_dim"]
        setattr(self, attr + "_feature_dim", feature_dim)

        feat_encoder_config = copy.deepcopy(feat_encoder)
        with omegaconf.open_dict(feat_encoder_config):
            feat_encoder_config.params.model_data_dir = self.config.model_data_dir
            feat_encoder_config.params.in_dim = feature_dim
        feat_model = build_image_encoder(feat_encoder_config, direct_features=False)

        setattr(self, attr + "_feature_dim", feat_model.out_dim)
        setattr(self, attr + "_feature_encoders", feat_model)

    def _init_feature_embeddings(self, attr: str):
        embedding_kwargs = self.config[attr + "_feature_embeddings"]["params"]
        setattr(
            self, attr + "_feature_embeddings_out_dim", embedding_kwargs["hidden_dim"]
        )
        assert (
            getattr(self, attr + "_feature_embeddings_out_dim")
            == self.text_embeddings_out_dim
        ), "dim1: {}, dim2: {}".format(
            getattr(self, attr + "_feature_embeddings_out_dim"),
            self.text_embeddings_out_dim,
        )

        feature_embedding = TwoBranchEmbedding(
            getattr(self, attr + "_feature_dim"), **embedding_kwargs
        )
        setattr(self, attr + "_feature_embeddings_list", feature_embedding)

    def _get_embeddings_attr(self, attr: str):
        embedding_attr1 = attr
        if hasattr(self, attr + "_embeddings_out_dim"):
            embedding_attr1 = attr + "_embeddings_out_dim"
        else:
            embedding_attr1 = attr + "_feature_embeddings_out_dim"

        return embedding_attr1

    def _init_combine_layer(self, attr1: str, attr2: str):
        multi_modal_combine_layer = BranchCombineLayer(
            getattr(self, self._get_embeddings_attr(attr1)),
            getattr(self, self._get_embeddings_attr(attr2)),
        )

        setattr(
            self,
            attr1 + "_" + attr2 + "_multi_modal_combine_layer",
            multi_modal_combine_layer,
        )

    def _init_classifier(self, combined_embedding_dim: int):
        # TODO: Later support multihead
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        params = self.config["classifier"].get("params")
        if params is None:
            params = {}

        self.classifier = ClassifierLayer(
            self.config.classifier.type,
            in_dim=combined_embedding_dim,
            out_dim=num_choices,
            **params
        )

    def _init_extras(self):
        self.inter_model = None

    def get_optimizer_parameters(self, config: DictConfig) -> List[Dict[str, Any]]:
        combine_layer = self.image_text_multi_modal_combine_layer
        params = [
            {"params": filter_grads(self.word_embedding.parameters())},
            {
                "params": filter_grads(
                    self.image_feature_embeddings_list.sga.parameters()
                )
            },
            {
                "params": filter_grads(
                    self.image_feature_embeddings_list.sga_pool.parameters()
                )
            },
            {
                "params": filter_grads(
                    self.image_feature_embeddings_list.cbn.parameters()
                ),
                "lr": (
                    config.optimizer.params.lr * config.training.encoder_lr_multiply
                ),
            },
            {"params": filter_grads(self.text_embeddings.parameters())},
            {"params": filter_grads(combine_layer.parameters())},
            {"params": filter_grads(self.classifier.parameters())},
            {"params": filter_grads(self.image_feature_encoders.parameters())},
        ]

        return params

    def get_mapping(self):
        mapping = [
            "word_embedding",
            "image_feature_embeddings_list_sga",
            "image_feature_embeddings_list_sga_pool",
            "image_feature_embeddings_list_cbn",
            "text_embeddings",
            "combine_layer",
            "classifier",
            "image_feature_encoders",
        ]
        return mapping

    def _get_classifier_input_dim(self):
        return self.image_text_multi_modal_combine_layer.out_dim

    def process_text_embedding(
        self, sample_list: Dict[str, Any], embedding_attr: str = "text_embeddings"
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Get "text" attribute in case of "text_embeddings" case
        # and "context" attribute in case of "context_embeddings"
        texts = getattr(sample_list, embedding_attr.split("_")[0])

        # Get embedding models
        text_embedding_model = getattr(self, embedding_attr)

        # TODO: Move this logic inside
        if isinstance(text_embedding_model, PreExtractedEmbedding):
            text_embedding_total = text_embedding_model(sample_list.question_id)
        else:
            text_embedding_total, text_embedding_vec = text_embedding_model(
                texts, sample_list.text_mask
            )

        return text_embedding_total, text_embedding_vec

    def process_feature_embedding(
        self,
        attr: str,
        sample_list: Dict[str, Any],
        text_embedding_total: torch.Tensor,
        text_embedding_vec: torch.Tensor,
        extra: list = [],
        batch_size_t: Optional[int] = None,
    ):
        batch_size_t = (
            sample_list.get_batch_size() if batch_size_t is None else batch_size_t
        )

        # Convert list of keys to the actual values
        if hasattr(sample_list, "image"):
            feature = sample_list.image

            feature_encoder = getattr(self, attr + "_feature_encoders")
            encoded_feature = feature_encoder(feature)
            b, c, h, w = encoded_feature.shape
            padded_feat = torch.zeros(
                (b, c, 32, 32), dtype=torch.float, device=encoded_feature.device
            )
            padded_feat[:, :, :h, :w] = encoded_feature
            encoded_feature = padded_feat
        else:
            feature = sample_list.image_feature_0

            feature_encoder = getattr(self, attr + "_feature_encoders")
            encoded_feature = feature_encoder(feature)

        feature_embedding = getattr(self, attr + "_feature_embeddings_list")
        feature_sga, feature_cbn = feature_embedding(
            encoded_feature,
            text_embedding_total,
            text_embedding_vec,
            None,
            sample_list.text_mask,
        )

        return feature_sga, feature_cbn

    def combine_embeddings(self, *args):
        feature_names = args[0]
        v1, v2, q = args[1]

        layer = "_".join(feature_names) + "_multi_modal_combine_layer"
        return getattr(self, layer)(v1, v2, q)

    def calculate_logits(self, joint_embedding: torch.Tensor, **kwargs):
        return self.classifier(joint_embedding)

    def forward(self, sample_list: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        sample_list.text_mask = sample_list.text.eq(0)
        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total, text_embedding_vec = self.process_text_embedding(
            sample_list
        )

        feature_sga, feature_cbn = self.process_feature_embedding(
            "image", sample_list, text_embedding_total, text_embedding_vec[:, 0]
        )

        joint_embedding = self.combine_embeddings(
            ["image", "text"], [feature_sga, feature_cbn, text_embedding_vec[:, 1]]
        )

        model_output = {"scores": self.calculate_logits(joint_embedding)}

        return model_output
