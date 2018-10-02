from pythia.tasks.core.base_task import BaseTask


class QATask(BaseTask):
    def __init__(self, datasets):
        super(QATask, self).__init__('qa', datasets)

    def _get_available_datasets(self):
        return [
            'VQADataset',
            'VizWizDataset'
        ]

    def _preprocess_item(self, item):
        return item
