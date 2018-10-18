from pythia.core.tasks.base_task import BaseTask
from pythia.core.registry import Registry


@Registry.register_task('qa')
class QATask(BaseTask):
    def __init__(self, opts={}):
        super(QATask, self).__init__('qa', opts)

    def _get_available_datasets(self):
        return [
            'vqa',
            'vizwiz'
        ]

    def _preprocess_item(self, item):
        return item
