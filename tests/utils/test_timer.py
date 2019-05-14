# Copyright (c) Facebook, Inc. and its affiliates.
import time
import unittest

from pythia.utils.timer import Timer


class TestUtilsTimer(unittest.TestCase):
    def test_get_current(self):
        timer = Timer()
        expected = "000ms"

        self.assertEqual(timer.get_current(), expected)

    def test_reset(self):
        timer = Timer()
        time.sleep(2)
        timer.reset()
        expected = "000ms"

        self.assertEqual(timer.get_current(), expected)

    def test_get_time_since_start(self):
        timer = Timer()
        time.sleep(2)
        expected = "02s "

        self.assertTrue(expected in timer.get_time_since_start())
