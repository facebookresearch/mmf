# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import registry
from pythia.tasks import BaseTask


@registry.register_task("captioning")
class CaptioningTask(BaseTask):
    def __init__(self):
        super(CaptioningTask, self).__init__("captioning")

    def _get_available_datasets(self):
        return ["coco"]

    def _preprocess_item(self, item):
        return item
