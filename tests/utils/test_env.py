# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib
import io
import os
import sys
import unittest

from mmf.common.registry import registry
from mmf.utils.configuration import get_mmf_env
from mmf.utils.env import import_user_module, setup_imports
from mmf.utils.general import get_mmf_root
from mmf_cli.run import run
from tests.test_utils import make_temp_dir, search_log


class TestUtilsEnvE2E(unittest.TestCase):
    def _delete_dirty_modules(self):
        for key in list(sys.modules.keys()):
            if key not in self._initial_modules:
                del sys.modules[key]

    def _sanitize_registry(self):
        registry.mapping["builder_name_mapping"].pop("always_one", None)
        registry.mapping["model_name_mapping"].pop("simple", None)
        registry.mapping["state"] = {}

    def _get_user_dir(self, abs_path=True):
        if abs_path:
            return os.path.join(get_mmf_root(), "..", "tests", "data", "user_dir")
        else:
            return os.path.join("tests", "data", "user_dir")

    def setUp(self):
        setup_imports()
        self._initial_modules = set(sys.modules)
        self._sanitize_registry()

    def tearDown(self):
        self._delete_dirty_modules()
        self._sanitize_registry()

    def _test_user_import_e2e(self, extra_opts=None):
        if extra_opts is None:
            extra_opts = []

        MAX_UPDATES = 50
        user_dir = self._get_user_dir()
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
            opts = opts + extra_opts
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

    def test_user_import_e2e(self):
        self._test_user_import_e2e()

    def test_cpu_evaluation_e2e(self):
        self._test_user_import_e2e(extra_opts=["evaluation.use_cpu=True"])

    def test_import_user_module_from_directory_absolute(self, abs_path=True):
        # Make sure the modules are not available first
        self.assertIsNone(registry.get_builder_class("always_one"))
        self.assertIsNone(registry.get_model_class("simple"))
        self.assertFalse("mmf_user_dir" in sys.modules)

        # Now, import and test
        user_dir = self._get_user_dir(abs_path)
        import_user_module(user_dir)
        self.assertIsNotNone(registry.get_builder_class("always_one"))
        self.assertIsNotNone(registry.get_model_class("simple"))
        self.assertTrue("mmf_user_dir" in sys.modules)
        self.assertTrue(user_dir in get_mmf_env("user_dir"))

    def test_import_user_module_from_directory_relative(self):
        self.test_import_user_module_from_directory_absolute(abs_path=False)
        user_dir = self._get_user_dir(abs_path=False)
        self.assertEqual(user_dir, get_mmf_env("user_dir"))

    def test_import_user_module_from_file(self):
        self.assertIsNone(registry.get_builder_class("always_one"))
        self.assertIsNone(registry.get_model_class("simple"))

        user_dir = self._get_user_dir()
        user_file = os.path.join(user_dir, "models", "simple.py")
        import_user_module(user_file)
        # Only model should be found and build should be none
        self.assertIsNone(registry.get_builder_class("always_one"))
        self.assertIsNotNone(registry.get_model_class("simple"))
        self.assertTrue("mmf_user_dir" in sys.modules)
        self.assertTrue(user_dir in get_mmf_env("user_dir"))
