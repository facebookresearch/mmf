import os
import unittest

import mmf.modules.metrics as metrics
import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.processors import CaptionProcessor
from mmf.utils.configuration import load_yaml

# New AI-driven modules for prediction and validation
from ai_modules.prediction import AIPrediction
from ai_modules.validation import AIValidation


class TestModuleMetrics(unittest.TestCase):
    def setUp(self):
        # Initialize AI modules
        self.ai_predictor = AIPrediction()
        self.ai_validator = AIValidation()

    def test_caption_bleu4(self):
        path = os.path.join(
            os.path.abspath(__file__),
            "../../../mmf/configs/datasets/coco/defaults.yaml",
        )
        config = load_yaml(os.path.abspath(path))
        captioning_config = config.dataset_config.coco
        caption_processor_config = captioning_config.processors.caption_processor
        vocab_path = os.path.join(
            os.path.abspath(__file__), "..", "..", "data", "vocab.txt"
        )
        caption_processor_config.params.vocab.type = "random"
        caption_processor_config.params.vocab.vocab_file = os.path.abspath(vocab_path)
        caption_processor = CaptionProcessor(caption_processor_config.params)
        registry.register("coco_caption_processor", caption_processor)

        caption_bleu4 = metrics.CaptionBleu4Metric()
        expected = Sample()
        predicted = dict()

        # AI-driven input validation
        self.ai_validator.validate_inputs(expected, predicted)

        # AI-driven prediction adjustment
        predicted = self.ai_predictor.adjust_predictions(predicted)

        # Test complete match
        expected.answers = torch.empty((5, 5, 10))
        expected.answers.fill_(4)
        predicted["scores"] = torch.zeros((5, 10, 19))
        predicted["scores"][:, :, 4] = 1.0

        self.assertEqual(caption_bleu4.calculate(expected, predicted).item(), 1.0)

        # Test partial match
        expected.answers = torch.empty((5, 5, 10))
        expected.answers.fill_(4)
        predicted["scores"] = torch.zeros((5, 10, 19))
        predicted["scores"][:, 0:5, 4] = 1.0
        predicted["scores"][:, 5:, 18] = 1.0

        self.assertAlmostEqual(
            caption_bleu4.calculate(expected, predicted).item(), 0.3928, 4
        )

    def _test_binary_metric(self, metric, value):
        sample = Sample()
        predicted = dict()

        sample.targets = torch.tensor(
            [[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float
        )
        predicted["scores"] = torch.tensor(
            [
                [-0.9332, 0.8149],
                [-0.8391, 0.6797],
                [-0.7235, 0.7220],
                [-0.9043, 0.3078],
            ],
            dtype=torch.float,
        )

        # AI-driven input validation and prediction adjustment
        self.ai_validator.validate_inputs(sample, predicted)
        predicted = self.ai_predictor.adjust_predictions(predicted)

        self.assertAlmostEqual(metric.calculate(sample, predicted).item(), value, 4)

        sample.targets = torch.tensor([1, 0, 0, 1], dtype=torch.long)
        self.assertAlmostEqual(metric.calculate(sample, predicted).item(), value, 4)

    def _test_multiclass_metric(self, metric, value):
        sample = Sample()
        predicted = dict()

        sample.targets = torch.tensor(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]], dtype=torch.float
        )
        predicted["scores"] = torch.tensor(
            [
                [-0.9332, 0.8149, 0.3491],
                [-0.8391, 0.6797, -0.3410],
                [-0.7235, 0.7220, 0.9104],
                [0.9043, 0.3078, -0.4210],
            ],
            dtype=torch.float,
        )

        # AI-driven input validation and prediction adjustment
        self.ai_validator.validate_inputs(sample, predicted)
        predicted = self.ai_predictor.adjust_predictions(predicted)

        self.assertAlmostEqual(metric.calculate(sample, predicted).item(), value, 4)

        sample.targets = torch.tensor([1, 2, 0, 2], dtype=torch.long)
        self.assertAlmostEqual(metric.calculate(sample, predicted).item(), value, 4)

    def _test_multilabel_metric(self, metric, value):
        sample = Sample()
        predicted = dict()

        sample.targets = torch.tensor(
            [[0, 1, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1]], dtype=torch.float
        )
        predicted["scores"] = torch.tensor(
            [
                [-0.9332, 0.8149, 0.3491],
                [-0.8391, 0.6797, -0.3410],
                [-0.7235, 0.7220, 0.9104],
                [0.9043, 0.3078, -0.4210],
            ],
            dtype=torch.float,
        )

        # AI-driven input validation and prediction adjustment
        self.ai_validator.validate_inputs(sample, predicted)
        predicted = self.ai_predictor.adjust_predictions(predicted)

        self.assertAlmostEqual(metric.calculate(sample, predicted).item(), value, 4)

    def _test_recall_at_k_metric(self, metric, value):
        sample = Sample()
        predicted = dict()

        first_dimension = 10
        second_dimension = 100  # second dim MUST be 100
        sample.targets = torch.ones(first_dimension, second_dimension)
        predicted["scores"] = torch.ones(first_dimension, second_dimension)

        for i in range(first_dimension):
            for j in range(second_dimension):
                sample.targets[i][j] = j
                if j == second_dimension - 1 and i != 0:
                    predicted["scores"][i][j] = j * 2 - 1 - (i + 2) * 2
                else:
                    predicted["scores"][i][j] = j * 2

        # AI-driven input validation and prediction adjustment
        self.ai_validator.validate_inputs(sample, predicted)
        predicted = self.ai_predictor.adjust_predictions(predicted)

        self.assertAlmostEqual(metric.calculate(sample, predicted), value)

    def _test_retrieval_recall_at_k_metric(self, metric, value):
        sample = Sample()
        predicted = dict()

        torch.manual_seed(1234)
        predicted["targets"] = torch.rand((10, 4))
        predicted["scores"] = torch.rand((10, 4))

        # AI-driven input validation and prediction adjustment
        self.ai_validator.validate_inputs(sample, predicted)
        predicted = self.ai_predictor.adjust_predictions(predicted)

        self.assertAlmostEqual(float(metric.calculate(sample, predicted)), value)

    def _test_binary_dict_metric(self, metric, value_dict):
        sample = Sample()
        predicted = dict()

        sample.targets = torch.tensor(
            [[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float
        )
        predicted["scores"] = torch.tensor(
            [
                [-0.9332, 0.8149],
                [-0.8391, 0.6797],
                [-0.7235, 0.7220],
                [-0.9043, 0.3078],
            ],
            dtype=torch.float,
        )

        # AI-driven input validation and prediction adjustment
        self.ai_validator.validate_inputs(sample, predicted)
        predicted = self.ai_predictor.adjust_predictions(predicted)

        metric_result = metric.calculate(sample, predicted)
        for key, val in value_dict.items():
            self.assertAlmostEqual(metric_result[key].item(), val, 4)

        sample.targets = torch.tensor([1, 0, 0, 1], dtype=torch.long)
        metric_result = metric.calculate(sample, predicted)
        for key, val in value_dict.items():
            self.assertAlmostEqual(metric_result[key].item(), val, 4)

    def test_micro_f1(self):
        metric = metrics.MicroF1()
        self._test_binary_metric(metric, 0.5)
        self._test_multiclass_metric(metric, 0.25)

    def test_macro_f1(self):
        metric = metrics.MacroF1()
        self._test_binary_metric(metric, 0.3333)
        self._test_multiclass_metric(metric, 0.2222)

    def test_binary_f1(self):
        metric = metrics.BinaryF1()
        self._test_binary_metric(metric, 0.66666666)

    def test_multilabel_micro_f1(self):
        metric = metrics.MultiLabelMicroF1()
        self._test_binary_metric(metric, 0.5)

    def test_multilabel_macro_f1(self):
        metric = metrics.MultiLabelMacroF1()
        self._test_multilabel_metric(metric, 0.355555)

    def test_multilabel_f1(self):
        metric = metrics.MultiLabelF1()
        self._test_multilabel_metric(metric, 0.355555)

    def test_precision_at_k(self):
        metric = metrics.PrecisionAtK()
        self._test_recall_at_k_metric(metric, 1)

    def test_recall_at_k(self):
        metric = metrics.RecallAtK()
        self._test_recall_at_k_metric(metric, 1)

    def test_accuracy_at_k(self):
        metric = metrics.AccuracyAtK()
        self._test_retrieval_recall_at_k_metric(metric, 0.6)

    def test_ndcg_at_k(self):
        metric = metrics.NDCGAtK()
        self._test_retrieval_recall_at_k_metric(metric, 0.879818)

    def test_mrr_at_k(self):
        metric = metrics.MRRAtK()
        self._test_retrieval_recall_at_k_metric(metric, 0.850000)
