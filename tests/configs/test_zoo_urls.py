# Copyright (c) Facebook, Inc. and its affiliates.
import re
import time
import typing
import unittest

from mmf.common.typings import DictConfig, DownloadableFileType
from mmf.utils.configuration import load_yaml
from mmf.utils.download import DownloadableFile, check_header
from omegaconf import OmegaConf
from tests.test_utils import skip_if_macos, skip_if_no_network


class TestConfigsForKeys(unittest.TestCase):
    def _recurse_on_config(self, config: DictConfig, callback: typing.Callable):
        if OmegaConf.is_list(config) and len(config) > 0 and "url" in config[0]:
            # Found the urls, let's test them
            for item in config:
                # First try making the DownloadableFile class to make sure
                # everything is fine
                download = DownloadableFile(**item)
                # Now, call the actual callback which will test specific scenarios
                callback(download)

        elif OmegaConf.is_dict(config):
            # Both version and resources should be present
            if "version" in config:
                self.assertIn("resources", config)
            if "resources" in config:
                self.assertIn("version", config)

            # Let's continue recursing
            for item in config:
                self._recurse_on_config(config[item], callback=callback)

    def _check_download(self, download: DownloadableFileType):
        # Check the actual header 3 times before failing
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

    def _check_sha256sum(self, download: DownloadableFileType):
        if download._hashcode is not None:
            matches = re.findall(r"^[A-Fa-f0-9]{64}$", download._hashcode)
            assert len(matches) == 1, f"{download._url} doesn't have a valid sha256sum"

    def _test_zoo(self, path: str, callback: typing.Callable):
        zoo_config = load_yaml(path)
        self._recurse_on_config(zoo_config, callback=callback)

    def _test_all_zoos(self, callback: typing.Callable):
        self._test_zoo("configs/zoo/datasets.yaml", callback=callback)
        self._test_zoo("configs/zoo/models.yaml", callback=callback)

    @skip_if_no_network
    @skip_if_macos
    def test_zoos(self):
        self._test_all_zoos(callback=self._check_download)

    def test_sha256sums(self):
        self._test_all_zoos(callback=self._check_sha256sum)
