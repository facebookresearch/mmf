# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
from mmf.common.meter import Meter
from mmf.common.report import Report
from mmf.common.sample import SampleList


class TestMeter(unittest.TestCase):
    def test_meter_update_from_report(self):
        meter = Meter()
        prepared_batch = SampleList(
            {"targets": torch.tensor([1, 2, 3, 4]), "dataset_type": "val"}
        )
        for idx in range(5):
            model_output = {
                "scores": torch.tensor([0, 1, 2, 3]),
                "losses": {"loss": float(idx)},
            }
            report = Report(prepared_batch, model_output)
            meter.update_from_report(report)

        self.assertEqual(meter.loss.global_avg, 2.0)
        self.assertEqual(meter.loss.avg, 2.0)
