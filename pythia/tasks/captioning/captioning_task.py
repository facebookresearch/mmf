from pythia.core.tasks.base_task import BaseTask
from pythia.core.registry import Registry


@Registry.register_task('captioning')
class CaptioningTask(BaseTask):
    def __init__(self, datasets):
        super(CaptioningTask, self).__init__('dialog', datasets)

    def _get_available_datasets(self):
        return [
            'coco'
        ]

    def _preprocess_item(self, item):
        return item
