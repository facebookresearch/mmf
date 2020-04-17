#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# Original taken from ParlAI https://git.io/JvjfS, this file has been
# adapted for MMF use cases.

"""
Utilities for downloading and building data.

These can be replaced if your particular file system does not support them.
"""

import datetime
import hashlib
import json
import os
import shutil
import time

import requests
import tqdm


class DownloadableFile:
    """
    A class used to abstract any file that has to be downloaded online.

    Originally taken from ParlAI, this file has been modified for MMF specific
    use cases.

    Any dataset/model that needs to download a file needs to have a list RESOURCES
    that have objects of this class as elements.

    The class automatically figures out if the file is from Google Drive.

    This class provides the following functionality:

    - Download a file from a URL / Google Drive
    - Decompress the file if compressed
    - Checksum for the downloaded file
    - Send HEAD request to validate URL or Google Drive link
    - If the file is present and checksum is same, it won't be redownloaded

    Raises:
        AssertionError: If while downloading checksum of the files fails.
    """

    GOOGLE_DRIVE_SUBSTR = "drive.google"

    def __init__(
        self, url, file_name, hashcode=None, compressed=True, delete_original=False
    ):
        """
        An object of this class needs to be created with:

        Args:
            url (string): URL or Google Drive id to download from
            file_name (string): File name that the file should be named
            hashcode (string, optional): SHA256 hashcode of the downloaded file.
                                         Defaults to None. Won't be checked if not
                                         passed.
            compressed (bool, optional): False if the file is not compressed.
                                         Defaults to True.
            delete_original (bool, optional): If compressed whether to delete original.
                                              Defaults to False.
        """
        self._url = url
        self._file_name = file_name
        self._hashcode = hashcode
        self._compressed = compressed
        self._from_google = self._url.find(self.GOOGLE_DRIVE_SUBSTR) != -1
        self._delete_original = delete_original

    def checksum(self, download_path):
        """
        Checksum on a given file.

        Args:
            download_path (string): path to the downloaded file.
        """
        if self._hashcode is None:
            print("[ Checksum not provided, skipping ]")
            return

        sha256_hash = hashlib.sha256()
        destination = os.path.join(download_path, self._file_name)

        if not os.path.isfile(destination):
            # File is not present, nothing to checksum
            return

        with open(destination, "rb") as f:
            print("[ Starting checksum for {}]".format(self._file_name))
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() != self._hashcode:
                # remove_dir(download_path)
                raise AssertionError(
                    f"[ Checksum for {self._file_name} from \n{self._url}\n"
                    "does not match the expected checksum. Please try again. ]"
                )
            else:
                print("[ Checksum successful for {}]".format(self._file_name))

    def download_file(self, download_path):
        # First test if the link is actually downloadable
        self.check_header()

        downloaded = False
        redownload = False

        try:
            self.checksum(download_path)
        except AssertionError:
            # File exists but checksum has changed. Will be redownloaded
            print("[ Checksum changed for {}. Redownloading")
            redownload = True

        if self._from_google:
            downloaded = download_from_google_drive(
                self._url,
                os.path.join(download_path, self._file_name),
                redownload=redownload,
            )
        else:
            downloaded = download(
                self._url, download_path, self._file_name, redownload=redownload
            )

        # If download actually happened, then only checksum again and decompress
        if downloaded:
            self.checksum(download_path)

            if self._compressed:
                decompress(download_path, self._file_name, self._delete_original)

    def check_header(self):
        """
        Performs a HEAD request to check if the URL / Google Drive ID is live.
        """
        session = requests.Session()
        if self._from_google:
            URL = "https://docs.google.com/uc?export=download"
            response = session.head(URL, params={"id": self._url}, stream=True)
        else:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) "
                + "AppleWebKit/537.36 (KHTML, like Gecko) "
                + "Chrome/77.0.3865.90 Safari/537.36"
            }
            response = session.head(self._url, allow_redirects=True, headers=headers)
        status = response.status_code
        session.close()

        assert status == 200, (
            "The url {} is broken. If this is not your own url,"
            + " please open up an issue on GitHub"
        ).format(self._url)


def built(path, version_string=None):
    """
    Check if '.built' flag has been set for that task.

    If a version_string is provided, this has to match, or the version
    is regarded as not built.

    Version_string are generally the dataset version + the date the file was
    last updated. If this doesn't match, dataset will be mark not built. This makes
    sure that if we update our features or anything else features are updated
    for the end user.
    """
    if version_string:
        fname = os.path.join(path, ".built.json")
        if not os.path.isfile(fname):
            return False
        else:
            with open(fname, "r") as read:
                text = json.load(read)
            return text.get("version", None) == version_string
    else:
        return os.path.isfile(os.path.join(path, ".built.json"))


def mark_done(path, version_string=None):
    """
    Mark this path as prebuilt.

    Marks the path as done by adding a '.built' file with the current timestamp
    plus a version description string if specified.

    Args:
        path (str): The file path to mark as built
        version_string (str): The version of this dataset
    """
    data = {}
    data["created_at"] = str(datetime.datetime.today())
    data["version"] = version_string
    with open(os.path.join(path, ".built.json"), "w") as f:
        json.dump(data, f)


def download(url, path, fname, redownload=True):
    """
    Download file using `requests`.

    If ``redownload`` is set to false, then will not download tar file again if it is
    present (default ``True``).

    Returns whether download actually happened or not
    """
    outfile = os.path.join(path, fname)
    download = not os.path.isfile(outfile) or redownload
    retry = 5
    exp_backoff = [2 ** r for r in reversed(range(retry))]

    pbar = None
    if download:
        print("[ Downloading: " + url + " to " + outfile + " ]")
        pbar = tqdm.tqdm(unit="B", unit_scale=True, desc="Downloading {}".format(fname))

    while download and retry >= 0:
        resume_file = outfile + ".part"
        resume = os.path.isfile(resume_file)
        if resume:
            resume_pos = os.path.getsize(resume_file)
            mode = "ab"
        else:
            resume_pos = 0
            mode = "wb"
        response = None

        with requests.Session() as session:
            try:
                header = (
                    {"Range": "bytes=%d-" % resume_pos, "Accept-Encoding": "identity"}
                    if resume
                    else {}
                )
                response = session.get(url, stream=True, timeout=5, headers=header)

                # negative reply could be 'none' or just missing
                if resume and response.headers.get("Accept-Ranges", "none") == "none":
                    resume_pos = 0
                    mode = "wb"

                CHUNK_SIZE = 32768
                total_size = int(response.headers.get("Content-Length", -1))
                # server returns remaining size if resuming, so adjust total
                total_size += resume_pos
                pbar.total = total_size
                done = resume_pos

                with open(resume_file, mode) as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                        if total_size > 0:
                            done += len(chunk)
                            if total_size < done:
                                # don't freak out if content-length was too small
                                total_size = done
                                pbar.total = total_size
                            pbar.update(len(chunk))
                    break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ):
                retry -= 1
                pbar.clear()
                if retry >= 0:
                    print("Connection error, retrying. (%d retries left)" % retry)
                    time.sleep(exp_backoff[retry])
                else:
                    print("Retried too many times, stopped retrying.")
            finally:
                if response:
                    response.close()
    if retry < 0:
        raise RuntimeWarning("Connection broken too many times. Stopped retrying.")

    if download and retry > 0:
        pbar.update(done - pbar.n)
        if done < total_size:
            raise RuntimeWarning(
                "Received less data than specified in "
                + "Content-Length header for "
                + url
                + ". There may be a download problem."
            )
        move(resume_file, outfile)

    if pbar:
        pbar.close()

    return download


def make_dir(path):
    """
    Make the directory and any nonexistent parent directories (`mkdir -p`).
    """
    # the current working directory is a fine path
    if path != "":
        os.makedirs(path, exist_ok=True)


def move(path1, path2):
    """
    Rename the given file.
    """
    shutil.move(path1, path2)


def remove_dir(path):
    """
    Remove the given directory, if it exists.
    """
    shutil.rmtree(path, ignore_errors=True)


def decompress(path, fname, delete_original=True):
    """
    Unpack the given archive file to the same directory.

    Args:
        path(str): The folder containing the archive. Will contain the contents.
        fname (str): The filename of the archive file.
        delete_original (bool, optional): If true, the archive will be deleted
                                          after extraction. Default to True.
    """
    print("Unpacking " + fname)
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if delete_original:
        os.remove(fullpath)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def download_from_google_drive(gd_id, destination, redownload=True):
    """
    Use the requests package to download a file from Google Drive.
    """
    download = not os.path.isfile(destination) or redownload

    URL = "https://docs.google.com/uc?export=download"

    with requests.Session() as session:
        response = session.get(URL, params={"id": gd_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            response.close()
            params = {"id": gd_id, "confirm": token}
            response = session.get(URL, params=params, stream=True)

        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        response.close()

    return download
