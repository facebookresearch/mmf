# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import shutil
import sys

if len(sys.argv) != 3:
    exit("Usage: python tools/rename_genome_file.py [inDir] [outDir]")

inDir = sys.argv[1]
outDir = sys.argv[2]

OUT_NAME = "COCO_genome_%012d.npy"

os.makedirs(outDir, exist_ok=True)

n = 0
print("BEGIN.....")
for file in os.listdir(inDir):
    if file.endswith(".npy"):
        n += 1
        if n % 5000 == 0:
            print("process %d files" % n)
        image_id = int(file.split(".")[0])
        out_name = OUT_NAME % image_id
        in_file = os.path.join(inDir, file)
        out_file = os.path.join(outDir, out_name)
        shutil.copy(in_file, out_file)

print("process total %d files" % n)
print("DONE.....")
