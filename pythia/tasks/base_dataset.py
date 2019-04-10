import torch
import tqdm

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

from pythia.common.losses import Loss
from pythia.common.meter import Meter
from pythia.common.sample import SampleList
from pythia.common.registry import registry
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

    def init_loss_and_metrics(self, config):
        self.writer = registry.get('writer')
        self.should_log = registry.get('config').get('should_log', True)

        task_metrics = config.get('metrics', [])
        if isinstance(task_metrics, str):
            task_metrics = task_metrics.split(',')

        self.meter = Meter(self._name, self._dataset_type, task_metrics)
        self.loss_fn = Loss(config['loss'])

        if type(config['loss']) == dict:
            self.loss_name = config['loss']['type']
        else:
            self.loss_name = config['loss']

    def load_item(self, idx):
        raise NotImplementedError

    def calculate_loss_and_metrics(self, *args, **kwargs):
        self._calculate_metrics(*args, **kwargs)
        return self._calculate_loss(*args, **kwargs)

    def _calculate_metrics(self, *args, **kwargs):
        self.meter(*args, **kwargs)

    def _calculate_loss(self, *args, **kwargs):
        self.last_loss = self.loss_fn(*args, **kwargs)
        return self.last_loss

    def reset_meters(self):
        self.meter.reset()

    def init_processors(self):
        if not hasattr(self.config, "processors"):
            return
        extra_params = {
            'data_root_dir': self.config.data_root_dir
        }
        for processor_key, processor_params in self.config.processors.items():
            setattr(self, processor_key,
                    Processor(processor_params, **extra_params))

    def try_fast_read(self):
        return

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

    def get_single_call_funcs(self):
        return ["report_metrics"]

    def report_metrics(self, report, loss=None, extra_info=None,
                       should_print=True):
        if not self.should_log:
            return

        if loss is None:
            loss = self.last_loss
        if should_print:
            log_string = self.meter.get_log_string(loss)
            if extra_info is not None:
                log_string += " " + extra_info
            self.writer.write(log_string)

        dataset_type = self.meter.get_dataset_type()

        scalars = {}
        for i in range(len(self.meter.meter_types)):
            meter_type = self.meter.meter_types[i]
            value = self.meter.meter_values[i]

            key = "%s_%s_%s" % (self._name, dataset_type, meter_type)
            scalars[key] = value

        scalars["%s_%s" % (self._name, self.loss_name)] = loss

        self.writer.add_scalars(scalars, registry.get('current_iteration'))

    def format_for_evalai(self, batch, answers):
        return []

    def verbose_dump(self, *args, **kwargs):
        return
