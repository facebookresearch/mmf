from pythia.core.tasks import BaseTask
from pythia.core.registry import Registry


@Registry.register_task('qa')
class QATask(BaseTask):
    def __init__(self):
        super(QATask, self).__init__('qa')

    def _get_available_datasets(self):
        return [
            'vqa2',
            'vizwiz'
        ]

    def _preprocess_item(self, item):
        return item
