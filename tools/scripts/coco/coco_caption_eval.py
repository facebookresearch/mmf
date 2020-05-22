# Copyright (c) Facebook, Inc. and its affiliates.

# The following script requires Java 1.8.0 and pycocotools installed.
# The pycocoevalcap can be installed with pip as
# pip install git+https://github.com/flauted/coco-caption.git@python23
# Original pycocoevalcap code is at https://github.com/tylin/coco-caption
# but has no python3 support yet.

import argparse
import json

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


class COCOEvalCap:
    """
    COCOEvalCap code is adopted from https://github.com/tylin/coco-caption
    """

    def __init__(self, img_ids, coco, coco_res):
        self.eval_imgs = []
        self.eval = dict()
        self.img_to_eval = dict()
        self.coco = coco
        self.coco_res = coco_res

    def evaluate(self):
        gts = self.coco
        res = self.coco_res

        # =================================================
        # Set up scorers
        # =================================================
        print("tokenization...")
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print("setting up scorers...")
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print("computing %s score..." % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.set_eval(sc, m)
                    self.set_img_to_eval_imgs(scs, gts.keys(), m)
                    print(f"{m}: {sc:0.3f}")
            else:
                self.set_eval(score, method)
                self.set_img_to_eval_imgs(scores, gts.keys(), method)
                print(f"{method}: {score:0.3f}")
        self.set_eval_imgs()

    def set_eval(self, score, method):
        self.eval[method] = score

    def set_img_to_eval_imgs(self, scores, img_ids, method):
        for img_id, score in zip(img_ids, scores):
            if img_id not in self.img_to_eval:
                self.img_to_eval[img_id] = dict()
                self.img_to_eval[img_id]["image_id"] = img_id
            self.img_to_eval[img_id][method] = score

    def set_eval_imgs(self):
        self.eval_imgs = [eval for img_id, eval in self.img_to_eval.items()]


def calculate_metrics(img_ids, dataset_gts, dataset_res):
    img_to_anns_gts = {id: [] for id in img_ids}
    for ann in dataset_gts["annotations"]:
        img_to_anns_gts[ann["image_id"]] += [ann]

    img_to_anns_res = {id: [] for id in img_ids}
    for ann in dataset_res["annotations"]:
        img_to_anns_res[ann["image_id"]] += [ann]

    eval_obj = COCOEvalCap(img_ids, img_to_anns_gts, img_to_anns_res)
    eval_obj.evaluate()
    return eval_obj.eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image captioning metrics")
    parser.add_argument("--reference_json", help="Path to reference captions json")
    parser.add_argument("--predicted_json", help="Path to predicted captions json")
    args = parser.parse_args()

    with open(args.reference_json) as f:
        captions = json.load(f)

    references = []
    img_ids = []

    for img in captions["images"]:
        if img["split"] == "test":
            for c in img["sentences"]:
                d = {}
                d["image_id"] = c["imgid"]
                img_ids.append(c["imgid"])
                d["caption"] = c["raw"]
                references.append(d)
    img_ids = list(set(img_ids))

    with open(args.predicted_json) as f:
        preds = json.load(f)

    dataset_gts = {"annotations": references}
    dataset_res = {"annotations": preds}
    print(calculate_metrics(img_ids, dataset_gts, dataset_res))
