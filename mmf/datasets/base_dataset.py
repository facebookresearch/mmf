# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.common.sample import SampleList
from mmf.utils.general import get_current_device
from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    """Base class for implementing a dataset. Inherits from PyTorch's Dataset class
    but adds some custom functionality on top. Processors mentioned in the
    configuration are automatically initialized for the end user.

    Args:
        dataset_name (str): Name of your dataset to be used a representative
            in text strings
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
        self._global_config = registry.get("config")
        self._device = get_current_device()
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

        from mmf.utils.build import build_processors

        extra_params = {"data_dir": self.config.data_dir}
        reg_key = f"{self._dataset_name}_{{}}"
        processor_dict = build_processors(
            self.config.processors, reg_key, **extra_params
        )
        for processor_key, processor_instance in processor_dict.items():
            setattr(self, processor_key, processor_instance)
            full_key = reg_key.format(processor_key)
            registry.register(full_key, processor_instance)

    def prepare_batch(self, batch):
        """
        Can be possibly overridden in your child class

        Prepare batch for passing to model. Whatever returned from here will
        be directly passed to model's forward function. Currently moves the batch to
        proper device.

        Args:
            batch (SampleList): sample list containing the currently loaded batch

        Returns:
            sample_list (SampleList): Returns a sample representing current
                batch loaded
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

    def format_for_prediction(self, report):
        return []

    def verbose_dump(self, *args, **kwargs):
        return

    def visualize(self, num_samples=1, *args, **kwargs):
        raise NotImplementedError(
            f"{self.dataset_name} doesn't implement visualize function"
        )
