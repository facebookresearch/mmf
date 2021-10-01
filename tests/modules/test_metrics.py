# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest

import mmf.modules.metrics as metrics
import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.processors import CaptionProcessor
from mmf.utils.configuration import load_yaml


class TestModuleMetrics(unittest.TestCase):
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
                # sample = [[0, 1, 2, ..., 99], [0, 1, ..., 99], ...]
                sample.targets[i][j] = j
                if j == second_dimension - 1 and i != 0:
                    # changes last value or 'chosen candidate'
                    # to a lower rank as i increases
                    # predicted = [[0, 2, 4, ..., 198], [0, 2, ..., 196, 191],
                    # [0, ..., 196, 189], [0, ..., 196, 187], ...]
                    predicted["scores"][i][j] = j * 2 - 1 - (i + 2) * 2
                else:
                    # predicted = [[0, 2, 4, ..., 198], [0, 2, ...], ...]
                    predicted["scores"][i][j] = j * 2

        self.assertAlmostEqual(metric.calculate(sample, predicted), value)

    def _test_retrieval_recall_at_k_metric(self, metric, value):
        sample = Sample()
        predicted = dict()

        torch.manual_seed(1234)
        predicted["targets"] = torch.rand((10, 4))
        predicted["scores"] = torch.rand((10, 4))

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

    def test_micro_f1_precision_recall(self):
        metric = metrics.MicroF1PrecisionRecall()
        self._test_binary_dict_metric(
            metric, {"f1": 0.5, "precision": 0.5, "recall": 0.5}
        )

    def test_macro_f1_precision_recall(self):
        metric = metrics.MacroF1PrecisionRecall()
        self._test_binary_dict_metric(
            metric, {"f1": 0.3333, "precision": 0.25, "recall": 0.5}
        )

    def test_binary_f1_precision_recall(self):
        metric = metrics.BinaryF1PrecisionRecall()
        self._test_binary_dict_metric(
            metric, {"f1": 0.66666666, "precision": 0.5, "recall": 1.0}
        )

    def test_macro_roc_auc(self):
        metric = metrics.MacroROC_AUC()
        self._test_binary_metric(metric, 0.5)
        self._test_multiclass_metric(metric, 0.2222)

    def test_micro_roc_auc(self):
        metric = metrics.MicroROC_AUC()
        self._test_binary_metric(metric, 0.5)
        self._test_multiclass_metric(metric, 0.34375)

    def test_binary_ap(self):
        metric = metrics.BinaryAP()
        self._test_binary_metric(metric, 0.75)

    def test_recall_at_precision_k(self):
        metric = metrics.RecallAtPrecisionK(50)
        self._test_binary_metric(metric, 1.0)

        metric = metrics.RecallAtPrecisionK(90)
        self._test_binary_metric(metric, 0.5)

        metric = metrics.RecallAtPrecisionK(110)
        self._test_binary_metric(metric, 0)

    def test_micro_ap(self):
        metric = metrics.MicroAP()
        self._test_binary_metric(metric, 0.642857)
        self._test_multiclass_metric(metric, 0.354166)

    def test_macro_ap(self):
        metric = metrics.MacroAP()
        self._test_binary_metric(metric, 0.6666666)
        self._test_multiclass_metric(metric, 0.3888888)

    def test_recall_at_1(self):
        metric = metrics.RecallAt1()
        self._test_recall_at_k_metric(metric, 0.1)

    def test_recall_at_5(self):
        metric = metrics.RecallAt5()
        self._test_recall_at_k_metric(metric, 0.3)

    def test_recall_at_10(self):
        metric = metrics.RecallAt10()
        self._test_recall_at_k_metric(metric, 0.8)

    def test_retrieval_recall_at_1(self):
        metric = metrics.RecallAt1_ret()
        self._test_retrieval_recall_at_k_metric(metric, 0.1)

    def test_retrieval_recall_at_5(self):
        metric = metrics.RecallAt5_ret()
        self._test_retrieval_recall_at_k_metric(metric, 0.4)

    def test_retrieval_recall_at_10(self):
        metric = metrics.RecallAt10_ret()
        self._test_retrieval_recall_at_k_metric(metric, 1.0)

    def test_accuracy_base(self):
        metric = metrics.Accuracy()

        torch.manual_seed(2)
        targets = torch.rand((25, 10))
        scores = torch.rand((25, 10))

        acc = metric.calculate({"targets": targets}, {"scores": scores})
        self.assertAlmostEqual(0.04, acc.item())

    def test_accuracy_base2(self):
        metric = metrics.Accuracy()

        torch.manual_seed(2)
        targets = torch.rand((25, 10))
        scores = torch.rand((25, 10))
        scores = torch.max(scores, 1)[1]

        acc = metric.calculate({"targets": targets}, {"scores": scores})
        self.assertAlmostEqual(0.04, acc.item())

    def test_accuracy_base3(self):
        metric = metrics.Accuracy()

        torch.manual_seed(2)
        targets = torch.rand((25, 10))
        targets = torch.max(targets, 1)[1]
        scores = torch.rand((25, 10))

        acc = metric.calculate({"targets": targets}, {"scores": scores})
        self.assertAlmostEqual(0.04, acc.item())

    def test_accuracy_top1(self):
        metric = metrics.TopKAccuracy(score_key="scores", k=1)

        torch.manual_seed(2)
        targets = torch.rand((25, 10))
        scores = torch.rand((25, 10))
        targets = targets.topk(1, 1, True, True)[1].t().squeeze()

        acc = metric.calculate({"targets": targets}, {"scores": scores})
        self.assertAlmostEqual(0.04, acc.item(), 1)

    def test_accuracy_top1_with_max(self):
        metric = metrics.TopKAccuracy(score_key="scores", k=1)

        torch.manual_seed(2)
        targets = torch.rand((25, 10))
        targets = torch.max(targets, 1)[1]
        scores = torch.rand((25, 10))

        acc = metric.calculate({"targets": targets}, {"scores": scores})
        self.assertAlmostEqual(0.04, acc.item(), 1)

    def test_accuracy_top5(self):
        metric = metrics.TopKAccuracy(score_key="scores", k=5)

        torch.manual_seed(2)
        targets = torch.rand((25, 10))
        targets = torch.max(targets, 1)[1]
        scores = torch.rand((25, 10))

        acc = metric.calculate({"targets": targets}, {"scores": scores})
        self.assertAlmostEqual(0.48, acc.item(), 1)
