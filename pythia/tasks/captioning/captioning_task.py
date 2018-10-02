from pythia.tasks.core.base_task import BaseTask


class CaptioningTask(BaseTask):
    def __init__(self, datasets):
        super(CaptioningTask, self).__init__('dialog', datasets)

    def _get_available_datasets(self):
        return [
            'COCOCaptionsDataset'
        ]

    def _preprocess_item(self, item):
        return item
