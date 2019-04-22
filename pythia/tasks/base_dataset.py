# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import tqdm
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

from pythia.common.registry import registry
from pythia.common.sample import SampleList
from pythia.tasks.processors import Processor


class BaseDataset(Dataset):
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
        raise NotImplementedError

    def get_item(self, idx):
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
        sample = self.get_item(idx)
        sample.dataset_type = self._dataset_type
        sample.dataset_name = self._name
        return sample

    def prepare_batch(self, batch):
        """
        Can be possibly overriden in your child class

        Prepare batch for passing to model. Whatever returned from here will
        be directly passed to model's forward function

        Parameters
        ----------
        batch: dict
            Dictionary containing information about the next
            sample in batched form

        Returns
        -------
        data: dict
            Contains variables in the following format
            'texts': The main text of the batch which can be a question in
            most of the cases
            'image_features': Image features for the current batch
            'image_dim': Max BBoxes for the images
            'contexts': Contains context relevant to current batch, in VisDial
            this will be the history of the dialog till now

        obs: tensor
            Tensor containing observations for the current batch
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
