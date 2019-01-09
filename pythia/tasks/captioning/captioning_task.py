from pythia.core.tasks import BaseTask
from pythia.core.registry import registry


@registry.register_task('captioning')
class CaptioningTask(BaseTask):
    def __init__(self):
        super(CaptioningTask, self).__init__('dialog')

    def _get_available_datasets(self):
        return [
            'coco'
        ]

    def _preprocess_item(self, item):
        return item
