from pythia.core.tasks import BaseTask
from pythia.core.registry import registry


@registry.register_task('dialog')
class DialogTask(BaseTask):
    def __init__(self):
        super(DialogTask, self).__init__('dialog')

    def _get_available_datasets(self):
        return [
            'visdial'
        ]

    def _preprocess_item(self, item):
        return item
