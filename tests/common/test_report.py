# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

import tests.test_utils as test_utils
import torch
from mmf.common.report import Report
from mmf.common.sample import SampleList


class TestReport(unittest.TestCase):
    def _build_report(self):
        tensor_a = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
        sample_list = SampleList()
        sample_list.add_field("a", tensor_a)
        model_output = {"scores": torch.rand(2, 2)}

        report = Report(sample_list, model_output)
        return report

    def test_report_copy(self):
        original_report = self._build_report()
        report_copy = original_report.copy()

        report_copy["scores"].zero_()

        self.assertFalse(
            test_utils.compare_tensors(report_copy["scores"], original_report["scores"])
        )

    def test_report_detach(self):
        report = self._build_report()
        report.a = report.a.float()
        report.a.requires_grad = True
        report.scores = report.a * 2
        self.assertTrue(report.scores.requires_grad)
        self.assertTrue(report.a.requires_grad)
        self.assertFalse(report.scores.is_leaf)
        self.assertTrue(report.a.is_leaf)
        report = report.detach()
        self.assertFalse(report.scores.requires_grad)
        self.assertFalse(report.a.requires_grad)
        self.assertTrue(report.scores.is_leaf)
        self.assertTrue(report.a.is_leaf)

    @test_utils.skip_if_no_cuda
    def test_to_device(self):
        report = self._build_report()
        self.assertFalse(report.a.is_cuda)
        self.assertFalse(report.scores.is_cuda)

        report = report.to("cuda")

        self.assertTrue(report.a.is_cuda)
        self.assertTrue(report.scores.is_cuda)

        report = report.to("cpu", non_blocking=False)

        self.assertFalse(report.a.is_cuda)
        self.assertFalse(report.scores.is_cuda)

        report = report.to("cuda", fields=["scores"])

        self.assertFalse(report.a.is_cuda)
        self.assertTrue(report.scores.is_cuda)
