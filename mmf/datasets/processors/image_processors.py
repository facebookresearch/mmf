import collections

import torch
from omegaconf import OmegaConf
from torchvision import transforms

from mmf.common.registry import registry
from mmf.datasets.processors.processors import BaseProcessor


@registry.register_processor("torchvision_transforms")
class TorchvisionTransforms(BaseProcessor):
    def __init__(self, config, *args, **kwargs):
        transform_params = config.transforms
        assert OmegaConf.is_dict(transform_params) or OmegaConf.is_list(
            transform_params
        )
        if OmegaConf.is_dict(transform_params):
            transform_params = [transform_params]

        transforms_list = []

        for param in transform_params:
            if OmegaConf.is_dict(param):
                # This will throw config error if missing
                transform_type = param.type
                transform_param = param.get("params", OmegaConf.create({}))
            else:
                assert isinstance(param, str), (
                    "Each transform should either be str or dict containing "
                    + "type and params"
                )
                transform_type = param
                transform_param = OmegaConf.create([])

            transform = getattr(transforms, transform_type, None)
            # If torchvision doesn't contain this, check our registry if we
            # implemented a custom transform as processor
            if transform is None:
                transform = registry.get_processor_class(transform_type)
            assert (
                transform is not None
            ), f"torchvision.transforms has no transform {transform_type}"

            # https://github.com/omry/omegaconf/issues/248
            transform_param = OmegaConf.to_container(transform_param)
            # If a dict, it will be passed as **kwargs, else a list is *args
            if isinstance(transform_param, collections.abc.Mapping):
                transform_object = transform(**transform_param)
            else:
                transform_object = transform(*transform_param)

            transforms_list.append(transform_object)

        self.transform = transforms.Compose(transforms_list)

    def __call__(self, x):
        # Support both dict and normal mode
        if isinstance(x, collections.abc.Mapping):
            x = x["image"]
            return {"image": self.transform(x)}
        else:
            return self.transform(x)


@registry.register_processor("GrayScaleTo3Channels")
class GrayScaleTo3Channels(BaseProcessor):
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, x):
        if isinstance(x, collections.abc.Mapping):
            x = x["image"]
            return {"image": self.transform(x)}
        else:
            return self.transform(x)

    def transform(self, x):
        assert isinstance(x, torch.Tensor)
        # Handle grayscale, tile 3 times
        if x.size(0) == 1:
            x = torch.cat([x] * 3, dim=0)
        return x
