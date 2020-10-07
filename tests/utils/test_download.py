# Copyright (c) Facebook, Inc. and its affiliates.

import contextlib
import os
import tempfile
import unittest
from io import StringIO
from unittest import mock

import mmf.utils.download as download
import tests.test_utils as test_utils


TEST_DOWNLOAD_URL = (
    "https://dl.fbaipublicfiles.com/mmf/data/tests/visual_entailment_small.zip"
)
TEST_DOWNLOAD_SHASUM = (
    "e5831397710b71f58a02c243bb6e731989c8f37ef603aaf3ce18957ecd075bf5"
)


class TestUtilsDownload(unittest.TestCase):
    @test_utils.skip_if_no_network
    @test_utils.skip_if_macos
    def test_download_file_class(self):
        # Test normal scenario
        resource = download.DownloadableFile(
            TEST_DOWNLOAD_URL,
            "visual_entailment_small.zip",
            hashcode=TEST_DOWNLOAD_SHASUM,
            compressed=True,
        )

        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(StringIO()):
                resource.download_file(d)
            self.assertTrue(os.path.exists(os.path.join(d, "visual_entailment_small")))
            self.assertTrue(
                os.path.exists(
                    os.path.join(d, "visual_entailment_small", "db", "train.jsonl")
                )
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(
                        d,
                        "visual_entailment_small",
                        "features",
                        "features.lmdb",
                        "data.mdb",
                    )
                )
            )
            self.assertTrue(
                os.path.exists(os.path.join(d, "visual_entailment_small.zip"))
            )

        # Test when checksum fails
        resource = download.DownloadableFile(
            TEST_DOWNLOAD_URL,
            "visual_entailment_small.zip",
            hashcode="some_random_string",
            compressed=True,
        )

        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(StringIO()):
                self.assertRaises(AssertionError, resource.download_file, d)

        # Test when not compressed
        resource = download.DownloadableFile(
            TEST_DOWNLOAD_URL,
            "visual_entailment_small.zip",
            hashcode=TEST_DOWNLOAD_SHASUM,
            compressed=False,
        )

        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(StringIO()):
                resource.download_file(d)
            self.assertTrue(
                os.path.exists(os.path.join(d, "visual_entailment_small.zip"))
            )
            # Check already downloaded scenarios

            with mock.patch.object(resource, "checksum") as mocked:
                with contextlib.redirect_stdout(StringIO()):
                    resource.download_file(d)
                mocked.assert_called_once_with(d)

            with mock.patch("mmf.utils.download.download") as mocked:
                with contextlib.redirect_stdout(StringIO()):
                    resource.download_file(d)
                mocked.assert_called_once_with(
                    resource._url, d, resource._file_name, redownload=False
                )
            with mock.patch.object(resource, "checksum") as mocked:
                resource._hashcode = "some_random_string"
                with contextlib.redirect_stdout(StringIO()):
                    resource.download_file(d)
                self.assertTrue(mocked.call_count, 2)

            with mock.patch("mmf.utils.download.download") as mocked:
                resource._hashcode = "some_random_string"
                with contextlib.redirect_stdout(StringIO()):
                    self.assertRaises(AssertionError, resource.download_file, d)
                mocked.assert_called_once_with(
                    resource._url, d, resource._file_name, redownload=True
                )

        # Test delete original
        resource = download.DownloadableFile(
            TEST_DOWNLOAD_URL,
            "visual_entailment_small.zip",
            hashcode=TEST_DOWNLOAD_SHASUM,
            compressed=True,
            delete_original=True,
        )

        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(StringIO()):
                resource.download_file(d)
            self.assertFalse(
                os.path.exists(os.path.join(d, "visual_entailment_small.zip"))
            )

    def test_mark_done(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, ".built.json")
            self.assertFalse(os.path.exists(path))
            download.mark_done(d, "0.1")
            self.assertTrue(os.path.exists(path))

            with open(path) as f:
                import json

                data = json.load(f)
                self.assertEqual(list(data.keys()), ["created_at", "version"])

    def test_built(self):
        with tempfile.TemporaryDirectory() as d:
            # First, test without built file
            self.assertFalse(download.built(d, "0.2"))
            download.mark_done(d, "0.1")
            # Test correct version
            self.assertTrue(download.built(d, "0.1"))
            # Test wrong version
            self.assertFalse(download.built(d, "0.2"))
