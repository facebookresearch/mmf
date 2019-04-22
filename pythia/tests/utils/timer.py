# Copyright (c) Facebook, Inc. and its affiliates.
import time
import unittest

from pythia.utils.timer import Timer


class TestUtilsTimer(unittest.TestCase):
    def test_get_current(self):
        timer = Timer()
        expected = "00:00:00"

        self.assertEqual(timer.get_current(), expected)

    def test_reset(self):
        timer = Timer()
        time.sleep(2)
        timer.reset()
        expected = "00:00:00"

        self.assertEqual(timer.get_current(), expected)

    def test_get_time_since_start(self):
        timer = Timer()
        time.sleep(2)
        expected = "00:00:02"

        self.assertEqual(timer.get_time_since_start(), expected)
