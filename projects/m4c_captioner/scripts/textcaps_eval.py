# Copyright (c) Facebook, Inc. and its affiliates.

import json

import numpy as np
import tools.scripts.coco.coco_caption_eval as coco_caption_eval


def print_metrics(res_metrics):
    print(res_metrics)
    keys = [
        "Bleu_1",
        "Bleu_2",
        "Bleu_3",
        "Bleu_4",
        "METEOR",
        "ROUGE_L",
        "SPICE",
        "CIDEr",
    ]
    print("\n\n**********\nFinal model performance:\n**********")
    for k in keys:
        print(k, ": %.1f" % (res_metrics[k] * 100))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--set", type=str, default="val")
    args = parser.parse_args()

    if args.set not in ["train", "val"]:
        raise Exception(
            "this script only supports TextCaps train and val set. "
            "Please use the EvalAI server for test set evaluation"
        )

    with open(args.pred_file) as f:
        preds = json.load(f)
    annotation_file = args.annotation_file
    imdb = np.load(annotation_file, allow_pickle=True)
    imdb = imdb[1:]

    gts = [
        {"image_id": info["image_id"], "caption": info["caption_str"]} for info in imdb
    ]
    preds = [{"image_id": p["image_id"], "caption": p["caption"]} for p in preds]
    imgids = list({g["image_id"] for g in gts})

    metrics = coco_caption_eval.calculate_metrics(
        imgids, {"annotations": gts}, {"annotations": preds}
    )

    print_metrics(metrics)
