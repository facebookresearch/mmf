# Copyright (c) Facebook, Inc. and its affiliates.
import os
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.distributed import gather_tensor_along_batch, object_to_byte_tensor
from torch import nn, Tensor


class DetectionCOCODataset(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "detection_coco"
        super().__init__(name, config, dataset_type, *args, **kwargs)
        self.dataset_name = name

        image_dir = self.config.images[self._dataset_type][imdb_file_index]
        self.image_dir = os.path.join(self.config.data_dir, image_dir)
        coco_json = self.config.annotations[self._dataset_type][imdb_file_index]
        self.coco_json = os.path.join(self.config.data_dir, coco_json)

        self.coco_dataset = torchvision.datasets.CocoDetection(
            self.image_dir, self.coco_json
        )

        self.postprocessors = {"bbox": PostProcess()}

    def __getitem__(self, idx):
        img, target = self.coco_dataset[idx]
        image_id = self.coco_dataset.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self._load_coco_annotations(
            img, target, load_attributes=self.config.load_attributes
        )
        transform_out = self.detection_image_and_target_processor(
            {"img": img, "target": target, "dataset_type": self._dataset_type}
        )
        img = transform_out["img"]
        target = transform_out["target"]

        current_sample = Sample()
        current_sample.image_id = torch.tensor(image_id, dtype=torch.long)
        current_sample.image = img
        current_sample.targets_enc = object_to_byte_tensor(
            target, max_size=self.config.max_target_enc_size
        )
        current_sample.orig_size = target["orig_size"].clone().detach()

        return current_sample

    def __len__(self):
        return len(self.coco_dataset)

    def format_for_prediction(self, report):
        # gather detection output keys across processes
        pred_boxes = gather_tensor_along_batch(report.pred_boxes)
        pred_logits = gather_tensor_along_batch(report.pred_logits)
        orig_size = gather_tensor_along_batch(report.orig_size)

        outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
        if hasattr(report, "attr_logits"):
            attr_logits = gather_tensor_along_batch(report.attr_logits)
            outputs["attr_logits"] = attr_logits

        image_ids = report.image_id.tolist()
        results = self.postprocessors["bbox"](outputs, orig_size)

        predictions = []
        for image_id, r in zip(image_ids, results):
            scores = r["scores"].tolist()
            labels = r["labels"].tolist()
            # convert boxes from xyxy to xywh
            xmin, ymin, xmax, ymax = r["boxes"].unbind(1)
            boxes_xywh = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
            boxes_xywh = boxes_xywh.tolist()

            # group the boxes by image_id for de-duplication in `on_prediction_end`
            # (duplication is introduced by DistributedSampler)
            predictions.append(
                (
                    image_id,
                    [
                        {
                            "image_id": image_id,
                            "category_id": labels[k],
                            "bbox": box_xywh,
                            "score": scores[k],
                        }
                        for k, box_xywh in enumerate(boxes_xywh)
                    ],
                )
            )
            if "attr_scores" in r:
                attr_scores = r["attr_scores"].tolist()
                attr_labels = r["attr_labels"].tolist()
                for k in range(len(boxes_xywh)):
                    predictions[-1][1][k]["attr_score"] = attr_scores[k]
                    predictions[-1][1][k]["attr_label"] = attr_labels[k]

        return predictions

    def on_prediction_end(self, predictions):
        # de-duplicate the predictions (duplication is introduced by DistributedSampler)
        prediction_dict = {image_id: entries for image_id, entries in predictions}

        unique_entries = []
        for image_id in sorted(prediction_dict):
            unique_entries.extend(prediction_dict[image_id])

        return unique_entries

    def _load_coco_annotations(self, image, target, load_attributes=False):
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        attributes = None
        if load_attributes:
            # load the attribute annotations in the Visual Genome dataset
            # following vqa-maskrcnn-benchmark, -1 will be used as ignore label
            # (https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark)
            MAX_ATTR_NUM = 16
            attributes = -torch.ones(len(classes), MAX_ATTR_NUM, dtype=torch.int64)
            for n_obj, obj in enumerate(anno):
                attributes[n_obj] = torch.as_tensor(
                    obj["attribute_ids_max16"], dtype=torch.int64
                )

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if attributes is not None:
            attributes = attributes[keep]

        target = {}
        target["boxes"] = boxes
        target["orig_boxes"] = boxes
        target["labels"] = classes
        if attributes is not None:
            target["attributes"] = attributes
        target["image_id"] = image_id
        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        target["area"] = area[keep]
        target["orig_area"] = target["area"]
        iscrowd = torch.tensor([obj.get("iscrowd", 0) for obj in anno])
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


class PostProcess(nn.Module):
    # Mostly copy-pasted from
    # https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor], target_sizes: Tensor):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        from mmf.utils.box_ops import box_cxcywh_to_xyxy

        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        if "attr_logits" in outputs:
            assert len(outputs["attr_logits"]) == len(results)
            attr_scores, attr_labels = outputs["attr_logits"].max(-1)
            for idx, r in enumerate(results):
                r["attr_scores"] = attr_scores[idx]
                r["attr_labels"] = attr_labels[idx]

        return results
