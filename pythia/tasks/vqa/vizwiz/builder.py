from pythia.tasks.vqa.vqa2 import VQA2Builder
from pythia.common.registry import registry
from .dataset import VizWizDataset


@registry.register_builder('vizwiz')
class VizWizBuilder(VQA2Builder):
    def __init__(self):
        super(VizWizBuilder, self).__init__()
        self.dataset_name = 'VizWiz'
        self.set_dataset_class(VizWizDataset)

    def update_registry_for_model(self, config):
        super(VizWizBuilder, self).update_registry_for_model(config)

        copy_type = False

        if 'copy_type' in self.opts:
            copy_type = self.opts['copy_type']

        if copy_type:
            config['context_max_len'] = self.opts['context_max_len']
            config['num_choices'] += config['context_max_len']
