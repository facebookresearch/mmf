from pythia.tasks.datasets.vizwiz.builder import VizWizBuilder
from pythia.core.registry import Registry
from .dataset import TextVQADataset


@Registry.register_builder('textvqa')
class TextVQABuilder(VizWizBuilder):
    def __init__(self):
        super(VizWizBuilder, self).__init__()
        self.dataset_name = 'TextVQA'
        self.set_dataset_class(TextVQADataset)
