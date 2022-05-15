# Copyright (c) Facebook, Inc. and its affiliates.

import gc
import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.sample import SampleList
from mmf.models.mmf_transformer import MMFTransformer, MMFTransformerModalityConfig
from mmf.models.transformers.heads.mlm import MLM
from mmf.modules.encoders import (
    EncoderFactory,
    IdentityEncoder,
    ImageEncoderFactory,
    ImageEncoderTypes,
    ResNet152ImageEncoder,
    TextEncoderFactory,
    TextEncoderTypes,
)
from mmf.utils.build import build_model
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports, teardown_imports
from omegaconf import OmegaConf
from tests.test_utils import skip_if_no_pytorchvideo

BERT_VOCAB_SIZE = 30255
ROBERTA_VOCAB_SIZE = 50265
XLM_ROBERTA_VOCAB_SIZE = 250002


class TestMMFTransformerTorchscript(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        self.model_name = "mmf_transformer"
        args = test_utils.dummy_args(model=self.model_name)
        configuration = Configuration(args)
        self.config = configuration.get_config()
        self.config.model_config[self.model_name].model = self.model_name

    def tearDown(self):
        teardown_imports()
        del self.config
        del self.model_name
        gc.collect()

    def test_load_save_finetune_model(self):
        model = build_model(self.config.model_config[self.model_name])
        self.assertTrue(test_utils.verify_torchscript_models(model))

    def test_finetune_bert_base(self):
        model = build_model(self.config.model_config[self.model_name])
        model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=BERT_VOCAB_SIZE
            )
        )

    def test_finetune_roberta_base(self):
        self.config.model_config[self.model_name]["transformer_base"] = "roberta-base"
        model = build_model(self.config.model_config[self.model_name])
        model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=ROBERTA_VOCAB_SIZE
            )
        )

    @test_utils.skip_if_no_network
    def test_finetune_xlmr_base(self):
        self.config.model_config[self.model_name][
            "transformer_base"
        ] = "xlm-roberta-base"
        model = build_model(self.config.model_config[self.model_name])
        model.eval()
        self.assertTrue(
            test_utils.compare_torchscript_transformer_models(
                model, vocab_size=XLM_ROBERTA_VOCAB_SIZE
            )
        )


class TestMMFTransformerConfig(unittest.TestCase):
    def setUp(self):
        setup_imports()

    def tearDown(self):
        teardown_imports()

    def test_mmft_from_params(self):
        modalities_config = [
            MMFTransformerModalityConfig(
                type="image",
                key="image",
                embedding_dim=256,
                position_dim=1,
                segment_id=0,
                encoder=IdentityEncoder.Config(),
            ),
            MMFTransformerModalityConfig(
                type="text",
                key="text",
                embedding_dim=768,
                position_dim=512,
                segment_id=1,
                encoder=IdentityEncoder.Config(),
            ),
        ]
        mmft = MMFTransformer.from_params(modalities=modalities_config, num_labels=2)
        mmft.build()

        config = OmegaConf.structured(
            MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        )
        self.assertIsNotNone(mmft)
        self.assertEqual(mmft.config, config)

    def test_mmf_from_params_encoder_factory(self):
        modalities_config = [
            MMFTransformerModalityConfig(
                type="image",
                key="image",
                embedding_dim=256,
                position_dim=1,
                segment_id=0,
                encoder=ImageEncoderFactory.Config(type=ImageEncoderTypes.identity),
            ),
            MMFTransformerModalityConfig(
                type="text",
                key="text",
                embedding_dim=756,
                position_dim=512,
                segment_id=0,
                encoder=TextEncoderFactory.Config(type=TextEncoderTypes.identity),
            ),
        ]
        mmft = MMFTransformer.from_params(modalities=modalities_config, num_labels=2)
        mmft.build()

        config = OmegaConf.structured(
            MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        )
        self.assertIsNotNone(mmft)
        self.assertEqual(mmft.config, config)

    def test_mmft_pretrained(self):
        mmft = MMFTransformer.from_params(num_labels=2)
        self.assertIsNotNone(mmft)

    def test_mmft_from_build_model(self):
        modalities_config = [
            MMFTransformerModalityConfig(
                type="image",
                key="image",
                embedding_dim=256,
                position_dim=1,
                segment_id=0,
                encoder=ImageEncoderFactory.Config(
                    type=ImageEncoderTypes.resnet152,
                    params=ResNet152ImageEncoder.Config(pretrained=False),
                ),
            ),
            MMFTransformerModalityConfig(
                type="text",
                key="text",
                embedding_dim=756,
                position_dim=512,
                segment_id=1,
                encoder=TextEncoderFactory.Config(type=TextEncoderTypes.identity),
            ),
        ]
        config = MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        mmft = build_model(config)
        self.assertIsNotNone(mmft)


class TestMMFTransformer(unittest.TestCase):
    def setUp(self):
        test_utils.setup_proxy()
        setup_imports()
        self._image_modality_config = MMFTransformerModalityConfig(
            type="image",
            key="image",
            embedding_dim=256,
            position_dim=1,
            segment_id=0,
            encoder=ImageEncoderFactory.Config(type=ImageEncoderTypes.identity),
        )
        self._text_modality_config = MMFTransformerModalityConfig(
            type="text",
            key="text",
            embedding_dim=756,
            position_dim=128,
            segment_id=1,
            encoder=TextEncoderFactory.Config(type=TextEncoderTypes.identity),
        )

    def tearDown(self):
        teardown_imports()
        del self._image_modality_config
        del self._text_modality_config
        gc.collect()

    def test_one_dim_feature_preprocessing(self):
        modalities_config = [self._image_modality_config, self._text_modality_config]
        config = MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        mmft = build_model(config)

        sample_list = SampleList()
        sample_list.image = torch.rand(2, 256)
        sample_list.text = torch.randint(0, 512, (2, 128))

        transformer_input = mmft.preprocess_sample(sample_list)
        input_ids = transformer_input["input_ids"]
        self.assertEqual(input_ids["image"].dim(), 3)
        self.assertEqual(list(input_ids["image"].size()), [2, 1, 256])

        self.assertEqual(input_ids["text"].dim(), 2)
        self.assertEqual(list(input_ids["text"].size()), [2, 128])

        position_ids = transformer_input["position_ids"]
        test_utils.compare_tensors(position_ids["image"], torch.tensor([[0], [0]]))
        test_utils.compare_tensors(
            position_ids["text"], torch.arange(0, 128).unsqueeze(0).expand((2, 128))
        )

        masks = transformer_input["masks"]
        masks = mmft._infer_masks(sample_list, input_ids)
        test_utils.compare_tensors(masks["image"], torch.tensor([[1], [1]]))
        test_utils.compare_tensors(masks["text"], torch.ones((2, 128)).long())

        segment_ids = transformer_input["segment_ids"]
        test_utils.compare_tensors(segment_ids["image"], torch.tensor([[0], [0]]))
        test_utils.compare_tensors(segment_ids["text"], torch.ones((2, 128)).long())

        mlm_labels = transformer_input["mlm_labels"]
        test_utils.compare_tensors(
            mlm_labels["combined_labels"],
            torch.full((2, 129), dtype=torch.long, fill_value=-1),
        )

    def test_stacked_feature_preprocessing(self):
        self._text_modality_config.key = "body"
        second_text_modality_config = MMFTransformerModalityConfig(
            type="text",
            key="ocr",
            embedding_dim=756,
            position_dim=128,
            segment_id=2,
            encoder=TextEncoderFactory.Config(type=TextEncoderTypes.identity),
        )

        modalities_config = [
            self._image_modality_config,
            self._text_modality_config,
            second_text_modality_config,
        ]
        config = MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        mmft = build_model(config)

        sample_list = SampleList()
        sample_list.image = torch.rand(2, 256)
        # In stacked case, input_ids should represent all texts
        sample_list.input_ids = torch.randint(0, 512, (2, 2, 128))
        sample_list.lm_label_ids = torch.randint(-1, 30522, (2, 2, 128))
        lm_labels_sum = sample_list.lm_label_ids.sum().item()

        transformer_input = mmft.preprocess_sample(sample_list)
        self._compare_processed_for_multimodality(transformer_input, lm_labels_sum)

    def test_modality_key_preprocessing(self):
        self._text_modality_config.key = "body"
        second_text_modality_config = MMFTransformerModalityConfig(
            type="text",
            key="ocr",
            embedding_dim=756,
            position_dim=128,
            segment_id=2,
            encoder=TextEncoderFactory.Config(type=TextEncoderTypes.identity),
        )

        modalities_config = [
            self._image_modality_config,
            self._text_modality_config,
            second_text_modality_config,
        ]
        config = MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        mmft = build_model(config)

        sample_list = SampleList()
        sample_list.image = torch.rand(2, 256)
        sample_list.body = torch.randint(0, 512, (2, 128))
        sample_list.ocr = torch.randint(0, 512, (2, 128))
        sample_list.lm_label_ids = torch.randint(-1, 30522, (2, 128))
        lm_labels_sum = sample_list.lm_label_ids.sum().item() * 2

        transformer_input = mmft.preprocess_sample(sample_list)
        self._compare_processed_for_multimodality(transformer_input, lm_labels_sum)

    def _compare_processed_for_multimodality(self, transformer_input, lm_labels_sum=0):
        input_ids = transformer_input["input_ids"]
        self.assertEqual(input_ids["image"].dim(), 3)
        self.assertEqual(list(input_ids["image"].size()), [2, 1, 256])

        self.assertEqual(input_ids["body"].dim(), 2)
        self.assertEqual(list(input_ids["body"].size()), [2, 128])

        self.assertEqual(input_ids["ocr"].dim(), 2)
        self.assertEqual(list(input_ids["ocr"].size()), [2, 128])

        # Test specific modality keys case
        # Test encoder with resnet
        # Test input_mask case, test modality_mask case

        position_ids = transformer_input["position_ids"]
        test_utils.compare_tensors(position_ids["image"], torch.tensor([[0], [0]]))
        test_utils.compare_tensors(
            position_ids["body"], torch.arange(0, 128).unsqueeze(0).expand((2, 128))
        )
        test_utils.compare_tensors(
            position_ids["ocr"], torch.arange(0, 128).unsqueeze(0).expand((2, 128))
        )

        masks = transformer_input["masks"]
        test_utils.compare_tensors(masks["image"], torch.tensor([[1], [1]]))
        test_utils.compare_tensors(masks["body"], torch.ones((2, 128)).long())
        test_utils.compare_tensors(masks["ocr"], torch.ones((2, 128)).long())

        segment_ids = transformer_input["segment_ids"]
        test_utils.compare_tensors(segment_ids["image"], torch.tensor([[0], [0]]))
        test_utils.compare_tensors(segment_ids["body"], torch.ones((2, 128)).long())
        test_utils.compare_tensors(
            segment_ids["ocr"],
            torch.full((2, 128), dtype=torch.long, fill_value=2).long(),
        )

        mlm_labels = transformer_input["mlm_labels"]
        self.assertEqual(list(mlm_labels["combined_labels"].size()), [2, 257])
        # -2 is for image negative labels
        self.assertEqual(mlm_labels["combined_labels"].sum().item(), lm_labels_sum - 2)

    def test_custom_feature_and_mask_preprocessing(self):
        extra_modality = MMFTransformerModalityConfig(
            type="my_random_feature",
            key="my_random_feature",
            embedding_dim=128,
            position_dim=4,
            segment_id=3,
            encoder=EncoderFactory.Config(type="identity"),
        )

        modalities_config = [
            self._image_modality_config,
            self._text_modality_config,
            extra_modality,
        ]
        config = MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        mmft = build_model(config)

        sample_list = SampleList()
        sample_list.image = torch.rand(2, 256)
        sample_list.text = torch.randint(0, 512, (2, 128))
        sample_list.text_mask = torch.ones(2, 128)
        sample_list.text_mask[:, 70:] = 0
        sample_list.my_random_feature = torch.rand(2, 4, 128)
        sample_list.my_random_feature_mask = torch.ones(2, 4)
        sample_list.my_random_feature_mask[:, 3:] = 0

        transformer_input = mmft.preprocess_sample(sample_list)
        input_ids = transformer_input["input_ids"]
        self.assertEqual(input_ids["image"].dim(), 3)
        self.assertEqual(list(input_ids["image"].size()), [2, 1, 256])

        self.assertEqual(input_ids["text"].dim(), 2)
        self.assertEqual(list(input_ids["text"].size()), [2, 128])

        self.assertEqual(input_ids["my_random_feature"].dim(), 3)
        self.assertEqual(list(input_ids["my_random_feature"].size()), [2, 4, 128])

        position_ids = transformer_input["position_ids"]
        test_utils.compare_tensors(position_ids["image"], torch.tensor([[0], [0]]))
        test_utils.compare_tensors(
            position_ids["text"], torch.arange(0, 128).unsqueeze(0).expand((2, 128))
        )
        test_utils.compare_tensors(
            position_ids["my_random_feature"],
            torch.arange(0, 4).unsqueeze(0).expand((2, 4)),
        )

        masks = transformer_input["masks"]
        test_utils.compare_tensors(masks["image"], torch.tensor([[1], [1]]))
        self.assertEqual(masks["text"].sum().item(), 140)
        self.assertEqual(masks["my_random_feature"].sum().item(), 6)

        segment_ids = transformer_input["segment_ids"]
        test_utils.compare_tensors(segment_ids["image"], torch.tensor([[0], [0]]))
        test_utils.compare_tensors(segment_ids["text"], torch.ones((2, 128)).long())
        test_utils.compare_tensors(
            segment_ids["my_random_feature"],
            torch.full((2, 4), dtype=torch.long, fill_value=3).long(),
        )

    def test_preprocessing_with_resnet_encoder(self):
        self._image_modality_config = MMFTransformerModalityConfig(
            type="image",
            key="image",
            embedding_dim=2048,
            position_dim=1,
            segment_id=0,
            encoder=ImageEncoderFactory.Config(
                type=ImageEncoderTypes.resnet152,
                params=ResNet152ImageEncoder.Config(pretrained=False),
            ),
        )
        modalities_config = [self._image_modality_config, self._text_modality_config]
        config = MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        mmft = build_model(config)

        sample_list = SampleList()
        sample_list.image = torch.rand(2, 3, 224, 224)
        sample_list.text = torch.randint(0, 512, (2, 128))

        transformer_input = mmft.preprocess_sample(sample_list)

        input_ids = transformer_input["input_ids"]
        self.assertEqual(input_ids["image"].dim(), 3)
        self.assertEqual(list(input_ids["image"].size()), [2, 1, 2048])

        self.assertEqual(input_ids["text"].dim(), 2)
        self.assertEqual(list(input_ids["text"].size()), [2, 128])

        position_ids = transformer_input["position_ids"]
        test_utils.compare_tensors(position_ids["image"], torch.tensor([[0], [0]]))
        test_utils.compare_tensors(
            position_ids["text"], torch.arange(0, 128).unsqueeze(0).expand((2, 128))
        )

        masks = transformer_input["masks"]
        test_utils.compare_tensors(masks["image"], torch.tensor([[1], [1]]))
        test_utils.compare_tensors(masks["text"], torch.ones((2, 128)).long())

        segment_ids = transformer_input["segment_ids"]
        test_utils.compare_tensors(segment_ids["image"], torch.tensor([[0], [0]]))
        test_utils.compare_tensors(segment_ids["text"], torch.ones((2, 128)).long())

    @skip_if_no_pytorchvideo
    def test_preprocessing_with_mvit_encoder(self):
        encoder_config = OmegaConf.create(
            {
                "name": "pytorchvideo",
                "model_name": "mvit_base_32x3",
                "random_init": True,
                "drop_last_n_layers": 0,
                "pooler_name": "cls",
                "spatial_size": 224,
                "temporal_size": 8,
                "head": None,
                "embed_dim_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
                "atten_head_mul": [[1, 2.0], [3, 2.0], [14, 2.0]],
                "pool_q_stride_size": [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
                "pool_kv_stride_adaptive": [1, 8, 8],
                "pool_kvq_kernel": [3, 3, 3],
            }
        )
        self._image_modality_config = MMFTransformerModalityConfig(
            type="image",
            key="image",
            embedding_dim=768,
            position_dim=1,
            segment_id=0,
            encoder=encoder_config,
        )
        modalities_config = [self._image_modality_config, self._text_modality_config]
        config = MMFTransformer.Config(modalities=modalities_config, num_labels=2)
        mmft = build_model(config)

        sample_list = SampleList()
        sample_list.image = torch.rand((2, 3, 8, 224, 224))
        sample_list.text = torch.randint(0, 512, (2, 128))

        transformer_input = mmft.preprocess_sample(sample_list)
        input_ids = transformer_input["input_ids"]
        self.assertEqual(input_ids["image"].dim(), 3)
        self.assertEqual(list(input_ids["image"].size()), [2, 1, 768])

        self.assertEqual(input_ids["text"].dim(), 2)
        self.assertEqual(list(input_ids["text"].size()), [2, 128])

        position_ids = transformer_input["position_ids"]
        test_utils.compare_tensors(position_ids["image"], torch.tensor([[0], [0]]))
        test_utils.compare_tensors(
            position_ids["text"], torch.arange(0, 128).unsqueeze(0).expand((2, 128))
        )

        masks = transformer_input["masks"]
        test_utils.compare_tensors(masks["image"], torch.tensor([[1], [1]]))
        test_utils.compare_tensors(masks["text"], torch.ones((2, 128)).long())

        segment_ids = transformer_input["segment_ids"]
        test_utils.compare_tensors(segment_ids["image"], torch.tensor([[0], [0]]))
        test_utils.compare_tensors(segment_ids["text"], torch.ones((2, 128)).long())

    def test_tie_mlm_head_weight_to_encoder(self):
        self._text_modality_config = MMFTransformerModalityConfig(
            type="text",
            key="text",
            embedding_dim=768,
            position_dim=128,
            segment_id=0,
            encoder=TextEncoderFactory.Config(type=TextEncoderTypes.transformer),
        )
        heads = [MLM.Config()]
        modalities_config = [self._image_modality_config, self._text_modality_config]
        config = MMFTransformer.Config(
            heads=heads,
            modalities=modalities_config,
            num_labels=2,
            tie_weight_to_encoder="text",
        )
        mmft = build_model(config)

        test_utils.compare_tensors(
            mmft.heads[0].cls.predictions.decoder.weight,
            mmft.encoders["text"].embeddings.word_embeddings.weight,
        )
