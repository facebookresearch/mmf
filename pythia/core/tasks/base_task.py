import numpy as np
import torch

from torch.autograd import Variable
from torch.utils.data import Dataset

from pythia.core.registry import Registry


class BaseTask(Dataset):
    def __init__(self, task_name, **opts):
        self.opts = opts

        if 'datasets' not in opts:
            writer = Registry.get('writer')
            writer.write("No datasets attribute present for task: %s."
                         " Defaulting to all" % (task_name))
            datasets = "all"
        else:
            datasets = opts['datasets']

        if datasets is None:
            datasets = self._get_available_datasets()

        if type(datasets) == str:
            datasets = map(lambda x: x.strip(),
                           datasets.split(","))
        if len(datasets) == 0 and datasets[0] == "all":
            datasets = self._get_available_datasets()

        self.task_name = task_name
        self.given_datasets = datasets

    def load(self):
        self.datasets = []
        self.builders = []
        available_datasets = self._get_available_datasets()

        self.total_length = 0
        self.per_dataset_lengths = []
        self.num_datasets = 0

        for dataset in self.given_datasets:
            if dataset in available_datasets:
                builder_class = Registry.get_builder_class(dataset)

                if builder_class is None:
                    print("No builder class found for %s." % dataset)
                    continue
                builder_instance = builder_class()

                attributes = self.opts['dataset_attributes'][dataset]
                builder_instance.build(**attributes)
                dataset_instance = builder_instance.load(**self.opts)

                self.builders.append(builder_instance)
                self.datasets.append(dataset_instance)
                self.per_dataset_lengths.append(len(dataset_instance))
                self.total_length += len(dataset_instance)
            else:
                print("Dataset %s is not a valid dataset for task %s" %
                      (dataset, self.task_name))

        self.num_datasets = len(self.datasets)
        self.dataset_probablities = [1 for _ in range(self.num_datasets)]
        sampling = self.opts.get('dataset_size_proportional_sampling', None)

        if sampling is True:
            self.dataset_probablities = self.per_dataset_lengths
            self.dataset_probablities /= self.total_length

        self.change_dataset()

    def _get_available_datasets(self):
        """Set available datasets for this task here.
        Override in your child task class
        Temporary solution, later we will use decorators to easily register
        datasets with a task

        Returns
        -------
        type
            List - List of available datasets for this particular task
        """
        return []

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        idx = idx % self.per_dataset_lengths[self.dataset_choice]

        item = self.chosen_dataset[idx]

        return self._preprocess_item(item)

    def change_dataset(self):
        self.dataset_choice = np.random.choice(self.num_datasets, 1,
                                               p=self.dataset_probablities)
        self.chosen_dataset = self.datasets[self.dataset_choice]

    def prepare_batch(self, batch, use_cuda):
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

        if use_cuda:
            obs = obs.cuda()

        input_text_seqs = batch['texts']
        input_image_features = batch['image_feature_0']

        # TODO: Figure out what will be the default value here, which will be
        # linked to max_context_len
        input_contexts = batch.get('contexts', None)

        input_text_seqs = Variable(input_text_seqs.type(torch.LongTensor))
        input_image_features = Variable(input_image_features)
        input_contexts = Variable(input_contexts.type(torch.LongTensor))

        if use_cuda:
            input_text_seqs = input_text_seqs.cuda()
            input_image_features = input_image_features.cuda()
            input_contexts = input_contexts.cuda()

        image_feature_variables = [input_image_features]
        image_dim_variable = None

        if 'image_dim' in batch:
            image_dims = batch['image_dim']
            image_dim_variable = Variable(image_dims, requires_grad=False,
                                          volatile=False)

            if use_cuda:
                image_dim_variable = image_dim_variable.cuda()

        # check if more than 1 image_feat_batch
        i = 1
        image_feat_key = "image_features_%s"
        while image_feat_key % str(i) in batch:
            tmp_image_variable = Variable(batch[image_feat_key % str(i)])
            if use_cuda:
                tmp_image_variable = tmp_image_variable.cuda()
            image_feature_variables.append(tmp_image_variable)
            i += 1

        data = {
            'texts': input_text_seqs,
            'image_dim': image_dim_variable,
            'image_features': image_feature_variables,
            'context': input_contexts
        }

        return data, obs

    def calculate_loss(self, output, expected_output):
        return self.chosen_dataset.calculate_loss(output, expected_output)

    def report_metrics(self, loss, extra_info=None, should_print=True):
        self.chosen_dataset.report_metrics(loss, extra_info, should_print)

    def _preprocess_item(self, item):
        """Preprocess an item to be returned from __getitem__.
        Override in your child task class, so you have control on what you are
        returning

        Parameters
        ----------
        item : Object
            Object returned by a particular dataset

        Returns
        -------
        Object:
            Preprocessed item
        """
        raise NotImplementedError("This task doesn't implement preprocess_item"
                                  " method")

    def update_config_for_model(self, config):
        """
        Use this if there is some specific configuration required by model
        which must be inferred at runtime.
        """
        for builder in self.builders:
            builder.update_config_for_model(config)

    def init_args(self, parser):
        parser.add_argument_group('General Task Arguments')
        parser.add_argument('-dsp', '--dataset_size_proportional_sampling',
                            type=bool, default=0,
                            help="Pass if you want to sample from"
                            " dataset according to its size. Default: Equal "
                            " weighted sampling")

        self._init_args(parser)

    def _init_args(self, parser):
        """Override this function to add extra parameters to
        parser in your child task class.

        Parameters
        ----------
        parser : ArgumentParser
            Original parser object passed from the higher level classes like
            trainer

        Returns
        -------
        type
            Description of returned object.

        """
        for builder in self.builders:
            builder.init_args(parser)

    def clean_config(self, config):
        """
        Use this in case you want to clean the config you updated earlier
        in update_config_for_model
        """
        return NotImplemented("This task doesn't implement clean_config "
                              "method")
