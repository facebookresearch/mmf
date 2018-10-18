from pythia.core.tasks.base_task import BaseTask
from pythia.core.registry import Registry


@Registry.register_task('dialog')
class DialogTask(BaseTask):
    def __init__(self, datasets):
        super(DialogTask, self).__init__('dialog', datasets)

    def _get_available_datasets(self):
        return [
            'visdial'
        ]

    def _preprocess_item(self, item):
        return item
