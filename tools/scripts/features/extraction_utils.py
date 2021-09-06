# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import math
import os


def get_image_files(
    image_dir,
    exclude_list=None,
    partition=None,
    max_partition=None,
    start_index=0,
    end_index=None,
    output_folder=None,
):
    files = glob.glob(os.path.join(image_dir, "*.png"))
    files.extend(glob.glob(os.path.join(image_dir, "*.jpg")))
    files.extend(glob.glob(os.path.join(image_dir, "*.jpeg")))

    files = set(files)
    exclude = set()

    if os.path.exists(exclude_list):
        with open(exclude_list) as f:
            lines = f.readlines()
            for line in lines:
                exclude.add(line.strip("\n").split(os.path.sep)[-1].split(".")[0])
    output_ignore = set()
    if output_folder is not None:
        output_files = glob.glob(os.path.join(output_folder, "*.npy"))
        for f in output_files:
            file_name = f.split(os.path.sep)[-1].split(".")[0]
            output_ignore.add(file_name)

    for f in list(files):
        file_name = f.split(os.path.sep)[-1].split(".")[0]
        if file_name in exclude or file_name in output_ignore:
            files.remove(f)

    files = list(files)
    files = sorted(files)

    if partition is not None and max_partition is not None:
        interval = math.floor(len(files) / max_partition)
        if partition == max_partition:
            files = files[partition * interval :]
        else:
            files = files[partition * interval : (partition + 1) * interval]

    if end_index is None:
        end_index = len(files)

    files = files[start_index:end_index]

    return files


def chunks(array, chunk_size):
    for i in range(0, len(array), chunk_size):
        yield array[i : i + chunk_size], i
