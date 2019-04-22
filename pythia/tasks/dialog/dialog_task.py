# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import registry
from pythia.tasks import BaseTask


@registry.register_task("dialog")
class DialogTask(BaseTask):
    def __init__(self):
        super(DialogTask, self).__init__("dialog")

    def _get_available_datasets(self):
        return ["visdial"]

    def _preprocess_item(self, item):
        return item
