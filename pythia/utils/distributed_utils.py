# Inspired from maskrcnn_benchmark
from torch import distributed as dist


def synchronize(self):
    if not dist.is_nccl_available():
        return
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def get_rank():
    if not dist.is_nccl_available():
        return
    if not dist.is_initialized():
        return
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_world_size():
    if not dist.is_nccl_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
