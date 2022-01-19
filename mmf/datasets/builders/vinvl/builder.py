# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.vinvl.dataset import VinVLDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from omegaconf import open_dict


@registry.register_builder("vinvl")
class VinVLBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="vinvl", dataset_class=VinVLDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, dataset_type="train_val")
        self.dataset_class = VinVLDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/vinvl/defaults.yaml"

    def load(self, config, dataset_type, *args, **kwargs):
        """The VinVL dataset is a dataset that augments an existing
        dataset within MMF. VinVL requires unique inputs for
        finetuning and pretraining unsupported by general datasets.
        To enable this functionality on arbitrary datasets,
        the VinVL dataset contains a base dataset,
        and returns an augmented version of samples from the
        base dataset.
        For more details, read the VinVL dataset docstring.

        The Builder:
        This class is a builder for the VinVL dataset.
        As the VinVL dataset must be constructed with an instance to
        a base dataset, configured by the client in the VinVL configs
        yaml. This builder class instantiates 2 datasets, then
        passes the base dataset to the VinVL dataset instance.

        The VinVL config is expected to have the following stucture,
        ```yaml
        dataset_config:
            vinvl:
                base_dataset_name: vqa2
                label_map: <path to label map>
                base_dataset: ${dataset_config.vqa2}
                processors:
                    text_processor:
                        type: vinvl_text_tokenizer
                        params:
                            ...
        ```
        Where base_dataset is the yaml config for the base dataset
        in this example vqa2.
        And base_dataset_name is vqa2.

        Returns:
            VinVLDataset: Instance of the VinVLDataset class which contains
            an base dataset instance.
        """
        base_dataset_name = config.get("base_dataset_name", "vqa2")
        base_dataset_config = config.get("base_dataset", config)
        # instantiate base dataset
        # instantiate base dataser builder
        base_dataset_builder_class = registry.get_builder_class(base_dataset_name)
        base_dataset_builder_instance = base_dataset_builder_class()
        # build base dataset instance
        base_dataset_builder_instance.build_dataset(base_dataset_config)
        base_dataset = base_dataset_builder_instance.load_dataset(
            base_dataset_config, dataset_type
        )
        if hasattr(base_dataset_builder_instance, "update_registry_for_model"):
            base_dataset_builder_instance.update_registry_for_model(base_dataset_config)

        # instantiate vinvl dataset
        vinvl_text_processor = config["processors"]["text_processor"]
        with open_dict(base_dataset_config):
            base_dataset_config["processors"]["text_processor"] = vinvl_text_processor
            base_dataset_config["label_map"] = config["label_map"]

        vinvl_dataset = super().load(base_dataset_config, dataset_type, *args, **kwargs)
        vinvl_dataset.set_base_dataset(base_dataset)
        return vinvl_dataset
