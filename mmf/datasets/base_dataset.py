# Copyright (c) Facebook, Inc. and its affiliates.
from torch.utils.data.dataset import Dataset

from mmf.common.registry import registry
from mmf.common.sample import SampleList
from mmf.datasets.processors.processors import Processor


class BaseDataset(Dataset):
    """Base class for implementing a dataset. Inherits from PyTorch's Dataset class
    but adds some custom functionality on top. Instead of ``__getitem__`` you have to implement
    ``__getitem__`` here. Processors mentioned in the configuration are automatically initialized for
    the end user.

    Args:
        dataset_name (str): Name of your dataset to be used a representative in text strings
        dataset_type (str): Type of your dataset. Normally, train|val|test
        config (DictConfig): Configuration for the current dataset
    """

    def __init__(self, dataset_name, config, dataset_type="train", *args, **kwargs):
        super().__init__()
        if config is None:
            config = {}
        self.config = config
        self._dataset_name = dataset_name
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

    def __getitem__(self, idx):
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
            if not processor_params:
                continue
            reg_key = "{}_{}".format(self._dataset_name, processor_key)
            reg_check = registry.get(reg_key, no_warning=True)

            if reg_check is None:
                processor_object = Processor(processor_params, **extra_params)
                setattr(self, processor_key, processor_object)
                registry.register(reg_key, processor_object)
            else:
                setattr(self, processor_key, reg_check)

    def try_fast_read(self):
        return

    def prepare_batch(self, batch):
        """
        Can be possibly overridden in your child class

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

    @property
    def dataset_type(self):
        return self._dataset_type

    @property
    def name(self):
        return self._dataset_name

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, name):
        self._dataset_name = name

    def format_for_evalai(self, report):
        return []

    def verbose_dump(self, *args, **kwargs):
        return
