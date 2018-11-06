from pythia.tasks.datasets.vqa2.builder import VQA2Builder
from pythia.core.registry import Registry
from .dataset import VizWizDataset


@Registry.register_builder('vizwiz')
class VizWizBuilder(VQA2Builder):
    def __init__(self):
        super(VizWizBuilder, self).__init__()
        self.dataset_name = 'VizWiz'
        self.set_dataset_class(VizWizDataset)

    def update_config_for_model(self, config):
        super(VizWizBuilder, self).update_config_for_model(config)

        copy_type = False

        if 'copy_type' in self.opts:
            copy_type = self.opts['copy_type']

        if copy_type:
            config['context_max_len'] = self.opts['context_max_len']
            config['num_choices'] += config['context_max_len']
