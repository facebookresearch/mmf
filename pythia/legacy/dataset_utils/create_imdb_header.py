# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import datetime

from global_variables.global_variables import imdb_version


def create_header(dataset_name, has_answer, has_gt_layout):
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M")
    version = imdb_version
    header = dict(
        create_time=time,
        dataset_name=dataset_name,
        version=version,
        has_answer=has_answer,
        has_gt_layout=has_gt_layout,
    )
    return header
