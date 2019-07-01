:github_url: https://github.com/facebookresearch/pythia

################
Adding a dataset
################

This is a tutorial on how to add a new dataset to Pythia.

Pythia is agnostic to kind of datasets that can be added to it. On high level, adding a dataset requires 4 main components. 

- Dataset Builder
- Default Configuration
- Dataset Class
- Dataset's Metrics
- [Optional] Task specification

In most of the cases, you should be able to inherit one of the existing datasets for easy integration. Let's start from the dataset builder


Dataset Builder
===============

Builder creates and returns an instance of :class:`pythia.tasks.base_dataset.BaseDataset` which is inherited from ``torch.utils.data.dataset.Dataset``.
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
    from pythia.tasks.base_dataset_builder import BaseDatasetBuilder
    # Let's assume for now that we have a dataset class called CLEVRDataset
    from pythia.tasks.vqa.clevr.dataset import CLEVRDataset
    from pythia.utils.general import download_file, get_pythia_root

    @registry.register_builder("clevr")
    class CLEVRBuilder(BaseDatasetBuilder):
        DOWNLOAD_URL = "https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip"

        def __init__(self):
            # Init should call super().__init__ with the key for the dataset
            super().__init__("clevr")
            self.writer = registry.get("writer")

            # Assign the dataset class
            self.dataset_class = CLEVRDataset

        def _build(self, dataset_type, config):
            download_folder = os.path.join(
                get_pythia_root(), config.data_root_dir, config.data_folder
            )

            file_name = self.DOWNLOAD_URL.split("/")[-1]
            local_filename = os.path.join(download_folder, file_name)

            extraction_folder = os.path.join(download_folder, ".".join(file_name.split(".")[:-1]))
            self.data_folder = extraction_folder

            # Either if the zip file is already present or if there are some
            # files inside the folder we don't continue download process
            if os.path.exists(local_filename):
                return

            if os.path.exists(extraction_folder) and \
                len(os.listdir(extraction_folder)) != 0:
                return

            self.writer.write("Downloading the CLEVR dataset now")
            download_file(self.DOWNLOAD_URL, output_dir=download_folder)

            self.writer.write("Downloaded. Extracting now. This can take time.")
            with zipfile.ZipFile(local_filename, "r") as zip_ref:
                zip_ref.extractall(download_folder)


        def _load(self, dataset_type, config, *args, **kwargs):
            # Load the dataset using the CLEVRDataset class
            self.dataset = CLEVRDataset(
                dataset_type, config, data_folder=self.data_folder
            )
            return self.dataset

        def update_registry_for_model(self, config):
            # Register both vocab (question and answer) sizes to registry for easy access to the 
            # models. update_registry_for_model function if present is automatically called by 
            # pythia
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
            registry.register(
                self.dataset_name + "_num_final_outputs",
                self.dataset.answer_processor.get_vocab_size(),
            )

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
            dataset_attributes:
                # You can specify any attributes you want, and you will get them as attributes 
                # inside the config passed to the dataset. Check the Dataset implementation below.
                clevr:
                    # Where your data is stored
                    data_root_dir: ../data
                    data_folder: CLEVR_v1.0
                    # Any attribute that you require to build your dataset but are configurable
                    # For CLEVR, we have attributes that can be passed to vocab building class
                    build_attributes:
                        min_count: 1
                        split_regex: " "
                        keep:
                            - ";"
                            - ","
                        remove:
                            - "?"
                            - "."
                    processors:
                    # The processors will be assigned to the datasets automatically by Pythia
                    # For example if key is text_processor, you can access that processor inside
                    # dataset object using self.text_processor
                        text_processor:
                            type: vocab
                            params:
                                max_length: 10
                                vocab:
                                    type: random
                                    vocab_file: vocabs/clevr_question_vocab.txt
                            # You can also specify a processor here
                            preprocessor:
                                type: simple_sentence
                                params: {}
                        answer_processor:
                            # Add your processor for answer processor here
                            type: multi_hot_answer_from_vocab
                            params:
                                num_answers: 1
                                # Vocab file is relative to [data_root_dir]/[data_folder]
                                vocab_file: vocabs/clevr_answer_vocab.txt
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

For processors, check :class:`pythia.tasks.processors` to understand how to create a processor and different processors that are
already available in Pythia.

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

.. note:

    Actual implementation of the dataset might differ due to support for distributed training.

.. code-block:: python

    import os
    import json

    import numpy as np
    import torch

    from PIL import Image

    from pythia.common.registry import registry
    from pythia.common.sample import Sample
    from pythia.tasks.base_dataset import BaseDataset
    from pythia.utils.general import get_pythia_root
    from pythia.utils.text_utils import VocabFromText, tokenize


    class CLEVRDataset(BaseDataset):
        def __init__(self, dataset_type, config, data_folder=None, *args, **kwargs):
            super().__init__("clevr", dataset_type, config)
            self._data_folder = data_folder
            self._data_root_dir = os.path.join(get_pythia_root(), config.data_root_dir)

            if not self._data_folder:
                self._data_folder = os.path.join(self._data_root_dir, config.data_folder)

            if not os.path.exists(self._data_folder):
                raise RuntimeError(
                    "Data folder {} for CLEVR is not present".format(self._data_folder)
                )

            # Check if the folder was actually extracted in the subfolder
            if config.data_folder in os.listdir(self._data_folder):
                self._data_folder = os.path.join(self._data_folder, config.data_folder)

            if len(os.listdir(self._data_folder)) == 0:
                raise RuntimeError("CLEVR dataset folder is empty")

            self._load()

        def _load(self):
            self.image_path = os.path.join(self._data_folder, "images", self._dataset_type)

            with open(
                os.path.join(
                    self._data_folder,
                    "questions",
                    "CLEVR_{}_questions.json".format(self._dataset_type),
                )
            ) as f:
                self.questions = json.load(f)["questions"]
                self._build_vocab(self.questions, "question")
                self._build_vocab(self.questions, "answer")

        def __len__(self):
            # __len__ tells how many samples are there
            return len(self.questions)

        def _get_vocab_path(self, attribute):
            return os.path.join(
                self._data_root_dir, "vocabs",
                "{}_{}_vocab.txt".format(self._name, attribute)
            )

        def _build_vocab(self, questions, attribute):
            # This function builds vocab for questions and answers but not required for the
            # tutorial
            ...

        def get_item(self, idx):
            # Get item is like your normal __getitem__ in PyTorch Dataset. Based on id 
            # return a sample. Check VQA2Dataset implementation if you want to see how 
            # to do caching in Pythia
            data = self.questions[idx]

            # Each call to get_item from dataloader returns a Sample class object which
            # collated by our special batch collator to a SampleList which is basically
            # a attribute based batch in layman terms
            current_sample = Sample()

            question = data["question"]
            tokens = tokenize(question, keep=[";", ","], remove=["?", "."])
            
            # This processors are directly assigned as attributes to dataset based on the config
            # we created above
            processed = self.text_processor({"tokens": tokens})
            # Add the question as text attribute to the sample
            current_sample.text = processed["text"]

            processed = self.answer_processor({"answers": [data["answer"]]})
            # Now add answers and then the targets. We normally use "targets" for what
            # should be the final output from the model in Pythia
            current_sample.answers = processed["answers"]
            current_sample.targets = processed["answers_scores"]

            image_path = os.path.join(self.image_path, data["image_filename"])
            image = np.true_divide(Image.open(image_path).convert("RGB"), 255)
            image = image.astype(np.float32)
            # Process and add image as a tensor
            current_sample.image = torch.from_numpy(image.transpose(2, 0, 1))

            # Return your sample and Pythia will automatically convert it to SampleList before 
            # passing to the model
            return current_sample



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

            if output.dim() == 2:
                output = torch.max(output, 1)[1]

            # If more than 1
            if expected.dim() == 2:
                expected = torch.max(expected, 1)[1]

            correct = (expected == output.squeeze()).sum().float()
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
