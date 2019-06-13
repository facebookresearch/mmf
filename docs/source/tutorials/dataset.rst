:github_url: https://github.com/facebookresearch/pythia

################
Adding a dataset
################

This is a tutorial on how to add a new dataset to Pythia.

Pythia is agnostic to kind of datasets that can be added to it. On high level, adding a dataset requires 4 main components. 

- Dataset Builder
- Dataset Class
- Default Configuration
- Dataset's Metrics
- [Optional] Task specification

In most of the cases, you should be able to inherit one of the existing datasets for easy integration. Let's start from the dataset builder


Dataset Builder
===============

Builder creates and returned an instance of :class:`pythia.tasks.base_dataset.BaseDataset` which is inherited from ``torch.utils.data.dataset.Dataset``.
Any builder class in Pythia needs to be inherited from :class:`pythia.tasks.base_dataset_builder.BaseDatasetBuilder`. |BaseDatasetBuilder| requires
user to implement following methods after inheriting the class.

- ``__init__(self):``

Inside this function call super().__init__("name") where "name" should your dataset's name like "vqa2". 

- ``_load(self, dataset_type, config, *args, **kwargs)``

This function loads the dataset, builds an object of class inheriting |BaseDataset| which contains your dataset logic and returns it.

- ``_build(self, dataset_type, config, *args, **kwargs)``

This function actually builds the data required for initializing the dataset for the first time. For e.g. if you need to download some data for your dataset, this 
all should be done inside this function. 

Finally, you need to register your dataset builder with a key to registry using ``pythia.common.registry.registry.register_builder("key")``.

That's it, that's all you require for inheriting |BaseDatasetBuilder|.

Let's write down this using example of *CLEVR* dataset.

.. code-block:: python
    
    import json
    import math
    import os
    import zipfile

    from collections import Counter
    
    from pythia.common.registry import registry
    from pythia.common.constants import DATA_FOLDER
    from pythia.tasks.base_dataset_builder import BaseDatasetBuilder
    # Let's assume for now that we have a dataset class called CLEVRDataset
    from pythia.tasks.vqa.clevr.dataset import CLEVRDataset
    from pythia.utils.general import download_url, get_pythia_root


    @registry.register_builder("clevr")
    class CLEVRBuilder(BaseDatasetBuilder):
        DOWNLOAD_URL = "https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip"
        FOLDER_NAME = "CLEVR_v1.0"

        def __init__(self):
            # Init should call super().__init__ with the key for the dataset
            super().__init__("clevr")
        
            self.writer = registry.get("writer") 
            # Assign the dataset class
            self.dataset_class = CLEVRDataset
        
        def _build(self, dataset_type, config):
            download_folder = getattr(config, "download_folder", DATA_FOLDER)
            download_folder = os.path.join(download_folder, self.FOLDER_NAME)
            download_folder = os.path.join(get_pythia_root(), download_folder)

            self.download_folder = download_folder

            file_name = self.DOWNLOAD_URL.split("/")[-1]
            local_filename = os.path.join(self.download_folder, file_name)

            if os.path.exists(local_filename):
                return

            if len(os.listdir(self.download_folder)) != 0:
                return

            self.writer.write("Downloading the CLEVR dataset now")

            download_url(self.DOWNLOAD_URL, output_folder=self.download_folder)

            with zipfile.ZipFile(local_filename, "r") as zip_ref:
                zip_ref.extractall(self.download_folder)
                            
        def _load(self, dataset_type, config, *args, **kwargs):
            self.dataset = CLEVRDataset(dataset_type, config, download_folder=self.download_folder)

            return self.dataset


.. note:

    Actual implementation of the dataset might differ due to support for distributed training.



Dataset Class
=============

Next step is to actually build a dataset class which inherits |BaseDataset| so it can interact with PyTorch
dataloaders. Follow the steps below to inherit and create your dataset's class.

- Inherit :class:`pythia.tasks.base_dataset.BaseDataset`
- Implement ``__init__(self, dataset_type, config)``. Call parent's init using ``super().__init__("name", dataset_type, config)``
  where "name" is the string representing the name of your dataset.
- Implement ``get_item(self, idx)``, our replacement for normal ``__getitem__(self, idx)`` you would implement for a torch dataset. This needs to 
  return an object of class :class:Sample. 
- Implement ``__len__(self)`` method, which represents size of your dataset.
- [Optional] Implement ``load_item(self, idx)`` if you need to load something or do something else with data and then call it inside ``get_item``.



.. code-block:: python

    import os

    from pythia.common.registry import registry
    from pythia.common.sample import Sample
    from pythia.tasks.base_dataset import BaseDataset
    from pythia.utils.general import get_pythia_root
    from pythia.utils.text_utils import VocabFromText

    
    class CLEVRDataset(BaseDataset):
        FOLDER_NAME = "CLEVR_v1.0"

        def __init__(self, dataset_type, config, download_folder=None, *args, **kwargs):
            super().__init__("clevr", dataset_type, config)

            self.download_folder = download_folder

            if not self.download_folder:
                self.download_folder = os.path.join(get_pythia_root(), self.FOLDER_NAME)

            if len(os.listdir(self.download_folder)) == 0:
                raise RuntimeError("CLEVR dataset folder is empty")
            
            self._load()
        
        def _load(self):
            self.image_path = os.path.join(folder_name, "images", self._dataset_type)
            with open(
                os.path.join(
                    self.download_folder,
                    "questions",
                    "CLEVR_{}_questions.json".format(self._dataset_type),
                )
            ) as f:
                self.questions = json.load(f)["questions"]

            self.question_vocab = self._load_vocab(
                self.download_folder, self.questions, "question"
            )
            self.answer_vocab = self._load_vocab(
                self.download_folder, self.questions, "answer"
            )
        
         def __len__(self):
            return len(self.questions)

        def _load_vocab(self, self.download_folder, questions, attribute):
            ....
        
        def get_item(self, idx):
            data = self.questions[idx]
            
            # Each call to get_item from dataloader returns a Sample class object which
            # collated by our special batch collator to a SampleList which is basically
            # a attribute based batch in layman terms
            current_sample = Sample()

            question = data["question"]

            processed = self.text_processor(
                {"text": question}, delimiter=" ", keep=[";", ","], remove=["?", "."]
            )

            current_sample.text = processed["text"]

            processed = self.answer_processor(
                {"answers": data["answer"]}
            )

            current_sample.answers = processed["answers"]
            current_sample.targets = processed["answers_scores"]
            
            image_path = os.path.join(self.image_path, data["image_filename"])
            image = np.true_divide(Image.open(image_path).convert("RGB"), 255)
            image = image.astype(np.float32)
            current_sample.image = torch.from_numpy(image.transpose(2, 0, 1))

            return current_sample


Default Configuration
=====================

Some things to note about Pythia's configuration:

- Each dataset in Pythia has its own default configuration which is usually under this structure 
  ``pythia/commmon/defaults/configs/tasks/[task]/[dataset].yml`` where ``task`` is the task your dataset belongs to.
- These dataset configurations can be then included by the user in their end config using ``includes`` directive
- This allows easy multi-tasking and management of configurations and user can also override the default configurations
  easily in their own config

So, for CLEVR dataset also, we will need to create a default configuration. 

The config node is directly passed to your builder which you can then pass to your dataset for any configuration that you need
for building your dataset. 

Basic structure for a dataset configuration looks like below:

.. code-block:: yaml

    task_attributes:
        [task]:
            datasets:
            - [dataset]
            dataset_attributes:
                [dataset]:
                    ... your config here

.. note:

    ``processors`` in your dataset configuration are directly converted to attributes based on the key and are
    automatically initialized with parameters mentioned in the config.

Here, is a default configuration for CLEVR needed based on our dataset and builder class above:

.. code-block:: yaml

    task_attributes:
        vqa:
            datasets:
            - clevr
            dataset_size_proportional_sampling: true
            dataset_attributes:
                clevr:
                    data_root_dir: ../data
                    processors:
                        text_processor:
                            type: vocab
                            params:
                                max_length: 10
                                vocab:
                                    type: random
                                    vocab_file: CLEVR_v1.0/question_vocab.txt
                                preprocessor:
                                    type: simple_sentence
                                    params: {}
                        answer_processor:
                            type: vocab
                            params:
                                max_length: 1
                                vocab:
                                    type: random
                                    vocab_file: CLEVR_v1.0/answer_vocab.txt
                                preprocessor:
                                    type: simple_word
                                    params: {}
    training_parameters:
        monitored_metric: clevr_accuracy
        metric_minimize: false


Extra field that we have added here is ``training_parameters`` which specify the dataset specific training parameters and will 
be merged with the rest of the training parameters coming from user's config. Your metrics are normally stored in registry as
``[dataset]_[metric_key]``, so to monitor accuracy on CLEVR, you need to set it as ``clevr_accuracy`` and we need to maximize it,
we set ``metric_minimize`` to ``false``.

.. note:

    Since, in v0.3, models are expected to return the metrics, so these attributes will also need to be specified by the user
    in future based on the metrics they are optimizing. Thus, in future warnings, these will move to user configs for models.


Metrics
=======

For your dataset to be compatible out of the box, it is a good practice to also add the metrics your dataset requires.
All metrics for now go inside ``pythia/modules/metrics.py``. All metrics inherit |BaseMetric| and implement a function ``calculate``
with signature ``calculate(self, sample_list, model_output, *args, **kwargs)`` where ``sample_list`` (|SampleList|) is the current batch and
``model_output`` is a dict return by your model for current ``sample_list``. Normally, you should define the keys you want inside
``model_output`` and ``sample_list``. Finally, you should register your metric to registry using ``@registry.register_metric('[key]')``
where '[key]' is the key for your metric. Here is a sample implementation of accuracy metric used in CLEVR dataset:

.. code-block: python

    @registry.register_metric("accuracy")
    class Accuracy(BaseMetric):
        """Metric for calculating accuracy.

        **Key:** ``accuracy``
        """

        def __init__(self):
            super().__init__("accuracy")

        def calculate(self, sample_list, model_output, *args, **kwargs):
            """Calculate accuracy and return it back.

            Args:
                sample_list (SampleList): SampleList provided by DataLoader for
                                    current iteration
                model_output (Dict): Dict returned by model.

            Returns:
                torch.FloatTensor: accuracy.

            """
            output = model_output["scores"]
            expected = sample_list["targets"]
            output = torch.max(output, 1)[1]

            correct = (expected == output.squeeze()).sum()

            correct = correct
            total = len(expected)

            value = correct / total
            return value


[Optional] Task Specification
=============================

This optional step is required in case you are adding a new task type to Pythia. Check implementation of VQATask_ to understand an 
implementation of task specification. In most cases, you don't need to do this.

These are the common steps you need to follow when you are adding a dataset to Pythia.

.. |BaseDatasetBuilder| replace:: :class:`~pythia.tasks.base_dataset_builder.BaseDatasetBuilder`
.. |BaseDataset| replace:: :class:`~pythia.tasks.base_dataset.BaseDataset`
.. |SampleList| replace:: :class:`~pythia.common.sample.SampleList`
.. _VQATask: https://github.com/facebookresearch/pythia/blob/master/pythia/tasks/vqa/vqa_task.py
.. |BaseMetric| replace:: :class:`~pythia.modules.metrics.BaseMetric`
