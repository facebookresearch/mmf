from torch.utils.data import Dataset


class BaseTask(Dataset):
    def __init__(self, task_name, dataset_type):
        self.task_name = task_name
        self.dataset_type = dataset_type
        return

    def build(self, opts):
        raise NotImplementedError("This task has no build method")

    def load(self, opts):
        raise NotImplementedError("This task has no load method")

    def __len__(self):
        raise NotImplementedError("This task doesn't implement length method")

    def __getitem__(self, idx):
        raise NotImplementedError("This task doesn't implement getitem method")

    def prepare_batch(self, batch, use_cuda=False):
        '''
        Override in your child class

        Prepare batch for passing to model. Whatever returned from here will
        be directly passed to model's forward function
        '''
        return batch

    def update_config_for_model(self, config):
        '''
        Use this if there is some specific configuration required by model
        which must be inferred at runtime.
        '''
        raise NotImplementedError("This task doesn't implement config"
                                  " update method")

    def init_args(self, parser):
        raise NotImplementedError("This task doesn't implement args "
                                  " initialization method")
