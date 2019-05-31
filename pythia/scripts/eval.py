# Copyright (c) Facebook, Inc. and its affiliates.

# The following script requires Java 1.8.0 and pycocotools installed.
# The pycocoevalcap can be installed with pip as
# pip install git+https://github.com/flauted/coco-caption.git@python23
# Original pycocoevalcap code is at https://github.com/tylin/coco-caption
# but has no python3 support yet.

import json
import argparse
from builtins import dict
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


class COCOEvalCap:
    """
    COCOEvalCap code is adopted from https://github.com/tylin/coco-caption
    """

    def __init__(self, imgIds, coco, cocoRes):
        self.evalImgs = []
        self.eval = dict()
        self.imgToEval = dict()
        self.coco = coco
        self.cocoRes = cocoRes
        # self.params = {"image_id": imgIds}

    def evaluate(self):
        gts = self.coco
        res = self.cocoRes

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
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = dict()
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def calculate_metrics(imgIds, datasetGTS, datasetRES):
    imgToAnnsGTS = {id: [] for id in imgIds}
    for ann in datasetGTS["annotations"]:
        imgToAnnsGTS[ann["image_id"]] += [ann]

    imgToAnnsRES = {id: [] for id in imgIds}
    for ann in datasetRES["annotations"]:
        imgToAnnsRES[ann["image_id"]] += [ann]

    evalObj = COCOEvalCap(imgIds, imgToAnnsGTS, imgToAnnsRES)
    evalObj.evaluate()
    return evalObj.eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image captioning metrics")
    parser.add_argument("--reference_json", help="Path to reference captions json")
    parser.add_argument("--predicted_json", help="Path to predicted captions json")
    args = parser.parse_args()

    with open(args.reference_json, "r") as f:
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

    with open(args.predicted_json, "r") as f:
        preds = json.load(f)

    datasetGTS = {"annotations": references}
    datasetRES = {"annotations": preds}
    print(calculate_metrics(img_ids, datasetGTS, datasetRES))
