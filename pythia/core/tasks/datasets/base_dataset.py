import torch

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

from pythia.core.losses import Loss
from pythia.core.meter import Meter
from pythia.core.registry import Registry


class BaseDataset(Dataset):
    def __init__(self, name, config={}):
        super(BaseDataset, self).__init__()
        self.config = config
        self.name = name
        self.use_cuda = Registry.get('config')['use_cuda']

        self.text_vocab = Registry.get('vocabs.text_vocab')
        self.context_vocab = Registry.get('vocabs.context_vocab')

    def init_loss_and_metrics(self, config):
        self.writer = Registry.get('writer')
        self.should_log = Registry.get('config').get('should_log', True)

        task_metrics = config.get('metrics', [])
        if isinstance(task_metrics, str):
            task_metrics = task_metrics.split(',')

        self.meter = Meter(self.name, config['dataset_type'], task_metrics)
        self.loss_fn = Loss(config['loss'])

    def calculate_loss(self, output, expected_output, info):
        self.meter(output, expected_output)

        self.last_loss = self.loss_fn(output, expected_output, info)

        return self.last_loss

    def reset_meters(self):
        self.meter.reset()

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
        obs = batch['answers']
        obs = Variable(obs.type(torch.FloatTensor))

        input_text_seqs = batch['texts']
        input_image_features = batch['image_feature_0']

        # TODO: Figure out what will be the default value here, which will be
        # linked to max_context_len
        #
        # TODO: Find a better way to clean this mess
        input_contexts = batch.get('contexts', None)

        input_text_seqs = Variable(input_text_seqs.type(torch.LongTensor))
        input_image_features = Variable(input_image_features)

        if input_contexts is not None:
            input_contexts = Variable(input_contexts.type(torch.LongTensor))

        if self.use_cuda:
            obs = obs.cuda()
            input_text_seqs = input_text_seqs.cuda()
            input_image_features = input_image_features.cuda()

            if input_contexts is not None:
                input_contexts = input_contexts.cuda()

        image_feature_variables = [input_image_features]
        image_dim_variable = None
        context_dim_variable = None

        if 'image_dim' in batch:
            image_dims = batch['image_dim']
            image_dim_variable = Variable(image_dims, requires_grad=False,
                                          volatile=False)

            if self.use_cuda:
                image_dim_variable = image_dim_variable.cuda()

        if 'context_dim' in batch:
            context_dims = batch['context_dim']
            context_dim_variable = Variable(context_dims, requires_grad=False,
                                            volatile=False)

            if self.use_cuda:
                context_dim_variable = context_dim_variable.cuda()

        # check if more than 1 image_feat_batch
        i = 1
        image_feat_key = "image_feature_%s"
        while image_feat_key % str(i) in batch:
            tmp_image_variable = Variable(batch[image_feat_key % str(i)])
            if self.use_cuda:
                tmp_image_variable = tmp_image_variable.cuda()
            image_feature_variables.append(tmp_image_variable)
            i += 1

        data = {
            'texts': input_text_seqs,
            'image_features': image_feature_variables,
            'contexts': input_contexts,
            'info': {
                'dataset_name': self.name,
                'image_dim': image_dim_variable,
                'context_dim': context_dim_variable
            }
        }

        if 'attention_supervision' in batch:
            att_sups = batch['attention_supervision']
            att_sups = Variable(att_sups, requires_grad=False, volatile=False)

            if self.use_cuda:
                att_sups = att_sups.cuda()
            data['info']['attention_supervision'] = att_sups

        return data, obs

    def report_metrics(self, loss=None, extra_info=None,
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

            key = "%s_%s_%s" % (self.name, dataset_type, meter_type)
            scalars[key] = value
        self.writer.add_scalars(scalars, Registry.get('current_iteration'))

    def format_for_evalai(self, batch, answers):
        return []
