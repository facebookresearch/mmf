# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import unittest
import warnings
from io import StringIO

from mmf.common.registry import registry
from mmf.utils.configuration import Configuration
from mmf.utils.env import setup_imports
from tests.test_utils import dummy_args


class TestConfigsForKeys(unittest.TestCase):
    def setUp(self):
        setup_imports()

    def test_model_configs_for_keys(self):
        models_mapping = registry.mapping["model_name_mapping"]

        for model_key, model_cls in models_mapping.items():
            if model_cls.config_path() is None:
                warnings.warn(
                    (
                        "Model {} has no default configuration defined. "
                        + "Skipping it. Make sure it is intentional"
                    ).format(model_key)
                )
                continue

            with contextlib.redirect_stdout(StringIO()):
                args = dummy_args(model=model_key)
                configuration = Configuration(args)
                configuration.freeze()
                config = configuration.get_config()
                self.assertTrue(
                    model_key in config.model_config,
                    "Key for model {} doesn't exists in its configuration".format(
                        model_key
                    ),
                )

    def test_dataset_configs_for_keys(self):
        builder_name = registry.mapping["builder_name_mapping"]

        for builder_key, builder_cls in builder_name.items():
            if builder_cls.config_path() is None:
                warnings.warn(
                    (
                        "Dataset {} has no default configuration defined. "
                        + "Skipping it. Make sure it is intentional"
                    ).format(builder_key)
                )
                continue

            with contextlib.redirect_stdout(StringIO()):
                args = dummy_args(dataset=builder_key)
                configuration = Configuration(args)
                configuration.freeze()
                config = configuration.get_config()
                self.assertTrue(
                    builder_key in config.dataset_config,
                    "Key for dataset {} doesn't exists in its configuration".format(
                        builder_key
                    ),
                )
