from pythia.core.tasks import BaseTask
from pythia.core.registry import Registry


@Registry.register_task('vqa')
class VQATask(BaseTask):
    def __init__(self):
        super(VQATask, self).__init__('vqa')

    def _get_available_datasets(self):
        return [
            'vqa2',
            'vizwiz'
        ]

    def _preprocess_item(self, item):
        return item
