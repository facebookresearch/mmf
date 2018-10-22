from pythia.tasks.datasets.vqa2.builder import VQA2Builder
from pythia.core.registry import Registry
from .dataset import VizWizDataset


@Registry.register_builder('vizwiz')
class VizWizBuilder(VQA2Builder):
    def __init__(self):
        super(VizWizBuilder, self).__init__()
        self.dataset_name = 'VizWiz'
        self.set_dataset_class(VizWizDataset)
