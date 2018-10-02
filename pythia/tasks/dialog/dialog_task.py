from pythia.tasks.core.base_task import BaseTask


class DialogTask(BaseTask):
    def __init__(self, datasets):
        super(DialogTask, self).__init__('dialog', datasets)

    def _get_available_datasets(self):
        return [
            'VisualDialogDataset'
        ]

    def _preprocess_item(self, item):
        return item
