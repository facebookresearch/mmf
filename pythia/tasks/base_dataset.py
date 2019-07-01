# Copyright (c) Facebook, Inc. and its affiliates.
from torch.utils.data.dataset import Dataset

from pythia.common.registry import registry
from pythia.common.sample import SampleList
from pythia.tasks.processors import Processor


class BaseDataset(Dataset):
    """Base class for implementing a dataset. Inherits from PyTorch's Dataset class
    but adds some custom functionality on top. Instead of ``__getitem__`` you have to implement
    ``get_item`` here. Processors mentioned in the configuration are automatically initialized for
    the end user.

    Args:
        name (str): Name of your dataset to be used a representative in text strings
        dataset_type (str): Type of your dataset. Normally, train|val|test
        config (ConfigNode): Configuration for the current dataset
    """
    def __init__(self, name, dataset_type, config={}):
        super(BaseDataset, self).__init__()
        self.config = config
        self._name = name
        self._dataset_type = dataset_type
        self.writer = registry.get("writer")
        self._global_config = registry.get("config")
        self._device = registry.get("current_device")
        self.use_cuda = "cuda" in str(self._device)

    def load_item(self, idx):
        """
        Implement if you need to separately load the item and cache it.

        Args:
            idx (int): Index of the sample to be loaded.
        """
        return

    def get_item(self, idx):
        """
        Basically, __getitem__ of a torch dataset.

        Args:
            idx (int): Index of the sample to be loaded.
        """

        raise NotImplementedError

    def init_processors(self):
        if not hasattr(self.config, "processors"):
            return
        extra_params = {"data_root_dir": self.config.data_root_dir}
        for processor_key, processor_params in self.config.processors.items():
            reg_key = "{}_{}".format(self._name, processor_key)
            reg_check = registry.get(reg_key, no_warning=True)

            if reg_check is None:
                processor_object = Processor(processor_params, **extra_params)
                setattr(self, processor_key, processor_object)
                registry.register(reg_key, processor_object)
            else:
                setattr(self, processor_key, reg_check)

    def try_fast_read(self):
        return

    def __getitem__(self, idx):
        # TODO: Add warning about overriding
        """
        Internal __getitem__. Don't override, instead override ``get_item`` for your usecase.

        .. warning::

            DO NOT OVERRIDE in child class. Instead override ``get_item``.
        """
        sample = self.get_item(idx)
        sample.dataset_type = self._dataset_type
        sample.dataset_name = self._name
        return sample

    def prepare_batch(self, batch):
        """
        Can be possibly overriden in your child class

        Prepare batch for passing to model. Whatever returned from here will
        be directly passed to model's forward function. Currently moves the batch to
        proper device.

        Args:
            batch (SampleList): sample list containing the currently loaded batch

        Returns:
            sample_list (SampleList): Returns a sample representing current batch loaded
        """
        # Should be a SampleList
        if not isinstance(batch, SampleList):
            # Try converting to SampleList
            batch = SampleList(batch)
        batch = batch.to(self._device)
        return batch

    def format_for_evalai(self, report):
        return []

    def verbose_dump(self, *args, **kwargs):
        return
