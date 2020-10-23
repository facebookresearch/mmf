# Copyright (c) Facebook, Inc. and its affiliates.
import unittest

import torch
from mmf.trainers.core.device import TrainerDeviceMixin
from mmf.utils.general import get_current_device
from omegaconf import OmegaConf


class DeviceMock(TrainerDeviceMixin):
    def __init__(self, config):
        self.config = config


class TestDevice(unittest.TestCase):
    def test_current_device(self):
        config = {"training": {"seed": 1}, "distributed": {"init_method": None}}
        deviceMock = DeviceMock(OmegaConf.create(config))
        deviceMock.configure_seed()
        deviceMock.configure_device()
        device = get_current_device()
        if torch.cuda.is_available():
            self.assertEqual(device, "cuda:0")
        else:
            self.assertEqual(device, torch.device(type="cpu"))
