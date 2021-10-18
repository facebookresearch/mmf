# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from mmf.utils.distributed import is_main


try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


def save_xla_ckpt(ckpt, file_or_path):
    """
    Similar to xm.save, but only try to convert "model" and "optimizer" in an MMF
    checkpoint to CPU, since they hold PyTorch tensors. Other items like lr_scheduler
    often cannot be saved with xm.save due to its errors in handling mappingproxy.

    Only save on the global main process (which is different from the default behavior
    of xm.save that saves a checkpoint on each node).
    """
    should_write_data = is_main()

    is_full_ckpt = isinstance(ckpt, dict) and "model" in ckpt and "optimizer" in ckpt
    if is_full_ckpt:
        ckpt["model"] = xm._maybe_convert_to_cpu(
            ckpt["model"], convert=should_write_data
        )
        ckpt["optimizer"] = xm._maybe_convert_to_cpu(
            ckpt["optimizer"], convert=should_write_data
        )
    else:
        ckpt = xm._maybe_convert_to_cpu(ckpt, convert=should_write_data)

    if should_write_data:
        torch.save(ckpt, file_or_path)
    xm.rendezvous("mmf.utils.checkpoint.save_xla_ckpt")
