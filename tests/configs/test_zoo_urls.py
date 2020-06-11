# Copyright (c) Facebook, Inc. and its affiliates.
import time
import unittest

from omegaconf import OmegaConf

from mmf.utils.configuration import load_yaml
from mmf.utils.download import DownloadableFile, check_header
from tests.test_utils import skip_if_no_network


class TestConfigsForKeys(unittest.TestCase):
    def _test_zoo_for_keys(self, path):
        zoo_config = load_yaml(path)
        self._recurse_on_config(zoo_config)

    def _recurse_on_config(self, config):
        if OmegaConf.is_list(config) and len(config) > 0 and "url" in config[0]:
            # Found the urls, let's test them
            for item in config:
                # First try making the DownloadableFile class to make sure
                # everything is fine
                download = DownloadableFile(**item)
                # Now check the actual header 3 times before failing
                for i in range(3):
                    try:
                        check_header(download._url, from_google=download._from_google)
                        break
                    except AssertionError:
                        if i == 2:
                            raise
                        else:
                            # If failed, add a sleep of 5 seconds before retrying
                            time.sleep(2)

        elif OmegaConf.is_dict(config):
            # Both version and resources should be present
            if "version" in config:
                self.assertIn("resources", config)
            if "resources" in config:
                self.assertIn("version", config)

            # Let's continue recursing
            for item in config:
                self._recurse_on_config(config[item])

    @skip_if_no_network
    def test_zoos(self):
        self._test_zoo_for_keys("configs/zoo/datasets.yaml")
        self._test_zoo_for_keys("configs/zoo/models.yaml")
