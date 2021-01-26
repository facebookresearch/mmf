# Copyright (c) Facebook, Inc. and its affiliates.

from tests.configs.test_zoo import TestConfigs
from tests.test_utils import skip_if_macos, skip_if_no_network


class TestConfigsInternal(TestConfigs):
    @skip_if_no_network
    @skip_if_macos
    def test_zoos(self):
        self._test_all_zoos(callback=self._check_download, url_type="internal")

    def test_sha256sums(self):
        self._test_all_zoos(callback=self._check_sha256sum, url_type="internal")
