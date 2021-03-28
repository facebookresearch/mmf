# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.utils.general import retry_n


MAX_RETRIES = 9

logger = logging.getLogger()


class GLUEDataset(BaseDataset):
    DATASET_KEY_MAP = {
        "text_a": {
            "glue_mnli_mismatched": "premise",
            "glue_qnli": "question",
            "glue_sst2": "sentence",
            "glue_qqp": "question1",
        },
        "text_b": {
            "glue_mnli_mismatched": "hypothesis",
            "glue_qnli": "sentence",
            "glue_sst2": None,
            "glue_qqp": "question2",
        },
    }

    def __init__(self, config, dataset_type, imdb_idx):
        try:
            from datasets import load_dataset
        except ModuleNotFoundError:
            logger.error(
                "Please install 'datasets' library by running `pip install datasets`."
            )
            raise

        dataset_name = f"glue_{config.task}"
        super().__init__(dataset_name, config, dataset_type, imdb_idx)

        if dataset_type == "val":
            # datasets library uses validation as val set.
            dataset_type = "validation"

        # For MNLI-MM, set train to MNLI-M train
        task = config.task
        if config.task.startswith("mnli") and dataset_type == "train":
            task = "mnli"

        self.dataset = retry_n(
            MAX_RETRIES, load_dataset, "glue", task, split=dataset_type
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        annotation = self.dataset[idx]
        current_sample = Sample()
        text_processor_input = {
            "text_a": annotation[self.DATASET_KEY_MAP["text_a"][self.dataset_name]]
        }

        text_b = annotation.get(self.DATASET_KEY_MAP["text_b"][self.dataset_name], None)
        if text_b is not None:
            text_processor_input["text_b"] = text_b

        current_sample.update(self.text_processor(text_processor_input))
        current_sample.targets = torch.tensor(annotation["label"], dtype=torch.long)
        return current_sample


@registry.register_builder("glue_sst2")
@registry.register_builder("glue_mnli_mismatched")
@registry.register_builder("glue_qqp")
@registry.register_builder("glue_qnli")
class GLUEBuilder(MMFDatasetBuilder):
    def __init__(self, dataset_name="glue", dataset_class=GLUEDataset, *args, **kwargs):
        super().__init__(dataset_name, dataset_class)
        self.dataset_name = dataset_name
        self.set_dataset_class(dataset_class)

    @classmethod
    def config_path(cls):
        return "configs/datasets/glue/defaults.yaml"

    def build(self, *args, **kwargs):
        # Will be built automatically by datasets library
        return

    def load(self, config, dataset_type, *args, **kwargs):
        self.dataset_name = f"{self.dataset_name}_{config.task}"
        return super().load(config, dataset_type)
