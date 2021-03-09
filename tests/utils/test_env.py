# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib
import io
import os
import unittest

from mmf.utils.general import get_mmf_root
from tests.test_utils import make_temp_dir, search_log

from mmf_cli.run import run


class TestUtilsEnv(unittest.TestCase):
    def test_user_import(self):
        MAX_UPDATES = 50
        user_dir = os.path.join(get_mmf_root(), "..", "tests", "data", "user_dir")
        with make_temp_dir() as temp_dir:
            opts = [
                "model=simple",
                "run_type=train_val_test",
                "dataset=always_one",
                "config=configs/experiment.yaml",
                f"env.user_dir={user_dir}",
                "training.seed=1",
                "training.num_workers=3",
                f"training.max_updates={MAX_UPDATES}",
                f"env.save_dir={temp_dir}",
            ]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                run(opts)
            train_log = os.path.join(temp_dir, "train.log")
            log_line = search_log(
                train_log,
                search_condition=[
                    lambda x: x["progress"] == f"{MAX_UPDATES}/{MAX_UPDATES}",
                    lambda x: "best_val/always_one/accuracy" in x,
                ],
            )
            self.assertEqual(float(log_line["val/always_one/accuracy"]), 1)

            log_line = search_log(
                train_log,
                search_condition=[
                    lambda x: x["progress"] == f"{MAX_UPDATES}/{MAX_UPDATES}",
                    lambda x: "test/always_one/accuracy" in x,
                ],
            )
            self.assertEqual(float(log_line["test/always_one/accuracy"]), 1)
