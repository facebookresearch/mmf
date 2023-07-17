# Copyright (c) Facebook, Inc. and its affiliates.
# Inspired from maskrcnn_benchmark, fairseq
import contextlib
import logging
import os
import pickle
import socket
import subprocess
import warnings
from itertools import chain

import torch
from mmf.common.registry import registry
from torch import distributed as dist


try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


MAX_SIZE_LIMIT = 65533
BYTE_SIZE = 256
logger = logging.getLogger(__name__)


# copied from https://github.com/facebookresearch/vissl/blob/master/vissl/utils/
# distributed_gradients.py
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class XLAGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all TPU workers with support for backward propagation.
    """

    @staticmethod
    def forward(ctx, x, dim):
        ctx.dim = dim
        tensor_list = xm.all_gather(x.unsqueeze(dim), dim=dim)
        return tensor_list

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx.dim
        all_grad_output = xm.all_reduce(xm.REDUCE_SUM, grad_output)
        return all_grad_output.select(dim, xm.get_ordinal()), None


def synchronize(message="sync-workers"):
    if is_xla():
        xm.rendezvous(message)
    elif not dist.is_available():
        return
    if not dist.is_nccl_available():
        return
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def is_xla():
    # Cover none case as well
    return not (not registry.get("is_xla", no_warning=True))


def get_rank():
    if is_xla():
        return xm.get_ordinal()
    if not dist.is_available():
        return 0
    if not dist.is_nccl_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main():
    return is_master()


def is_master():
    return get_rank() == 0


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_xla():
        return xm.xrt_world_size()
    if not dist.is_available():
        return 1
    if not dist.is_nccl_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def broadcast_tensor(tensor, src=0):
    world_size = get_world_size()
    if world_size < 2:
        return tensor

    with torch.no_grad():
        if is_xla():
            tensor = xm.all_to_all(
                tensor.repeat([world_size] + [1] * tensor.dim()),
                split_dimension=0,
                concat_dimension=0,
                split_count=world_size,
            )[src]
        else:
            dist.broadcast(tensor, src=0)

    return tensor


def broadcast_scalar(scalar, src=0, device="cpu"):
    if get_world_size() < 2:
        return scalar
    scalar_tensor = torch.tensor(scalar).long().to(device)
    scalar_tensor = broadcast_tensor(scalar_tensor, src)
    return scalar_tensor.item()


def reduce_tensor(tensor):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if dist.get_rank() == 0:
            tensor = tensor.div(world_size)

    return tensor


def gather_tensor(tensor):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        tensor_list = []

        if is_xla():
            tensor_list = xm.all_gather(tensor)
            tensor_list = tensor_list.view(world_size, *tensor.size())
        else:
            for _ in range(world_size):
                tensor_list.append(torch.zeros_like(tensor))
            dist.all_gather(tensor_list, tensor)
            tensor_list = torch.stack(tensor_list, dim=0)
    return tensor_list


def gather_tensor_along_batch(tensor, dim=0):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    with torch.no_grad():
        tensor_list = []

        for _ in range(world_size):
            tensor_list.append(torch.zeros_like(tensor))

        dist.all_gather(tensor_list, tensor)
        tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list


def gather_tensor_along_batch_with_backward(tensor, dim=0):
    world_size = get_world_size()

    if world_size < 2:
        return tensor

    if is_xla():
        tensor_list = XLAGatherLayer.apply(tensor, dim)
        tensor_list = tensor_list.flatten(start_dim=dim, end_dim=dim + 1)
    else:
        tensor_list = GatherLayer.apply(tensor)
        tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list


def reduce_dict(dictionary):
    world_size = get_world_size()
    if world_size < 2:
        return dictionary

    with torch.no_grad():
        if len(dictionary) == 0:
            return dictionary

        keys, values = zip(*sorted(dictionary.items()))
        values = torch.stack(values, dim=0)

        if is_xla():
            values = xm.all_reduce("sum", [values], scale=1.0 / world_size)[0]
        else:
            dist.reduce(values, dst=0)
            if dist.get_rank() == 0:
                # only main process gets accumulated, so only divide by
                # world_size in this case
                values /= world_size
        reduced_dict = {k: v for k, v in zip(keys, values)}
    return reduced_dict


# Object byte tensor utilities have been adopted from
# https://github.com/pytorch/fairseq/blob/main/fairseq/distributed_utils.py
def object_to_byte_tensor(obj, max_size=4094):
    """
    Encode Python objects to PyTorch byte tensors
    """
    assert max_size <= MAX_SIZE_LIMIT
    byte_tensor = torch.zeros(max_size, dtype=torch.uint8)

    obj_enc = pickle.dumps(obj)
    obj_size = len(obj_enc)
    if obj_size > max_size:
        raise Exception(
            f"objects too large: object size {obj_size}, max size {max_size}"
        )

    byte_tensor[0] = obj_size // 256
    byte_tensor[1] = obj_size % 256
    byte_tensor[2 : 2 + obj_size] = torch.ByteTensor(list(obj_enc))
    return byte_tensor


def byte_tensor_to_object(byte_tensor, max_size=MAX_SIZE_LIMIT):
    """
    Decode PyTorch byte tensors to Python objects
    """
    assert max_size <= MAX_SIZE_LIMIT

    obj_size = byte_tensor[0].item() * 256 + byte_tensor[1].item()
    obj_enc = bytes(byte_tensor[2 : 2 + obj_size].tolist())
    obj = pickle.loads(obj_enc)
    return obj


def infer_init_method(config):
    if config.distributed.init_method is not None:
        return

    registry.register("is_xla", config.training.get("device", "cuda") == "xla")

    # support torch.distributed.launch
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        config.distributed.init_method = "env://"
        config.distributed.world_size = int(os.environ["WORLD_SIZE"])
        config.distributed.rank = int(os.environ["RANK"])
        config.distributed.no_spawn = True

    # we can determine the init method automatically for Slurm
    elif config.distributed.port > 0:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                config.distributed.init_method = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"),
                    port=config.distributed.port,
                )
                nnodes = int(os.environ.get("SLURM_NNODES"))
                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get("SLURM_NTASKS"))
                    nnodes = int(os.environ.get("SLURM_NNODES"))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)
                if ntasks_per_node == 1:
                    assert config.distributed.world_size % nnodes == 0
                    gpus_per_node = config.distributed.world_size // nnodes
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    config.distributed.rank = node_id * gpus_per_node
                else:
                    assert ntasks_per_node == config.distributed.world_size // nnodes
                    config.distributed.no_spawn = True
                    config.distributed.rank = int(os.environ.get("SLURM_PROCID"))
                    config.device_id = int(os.environ.get("SLURM_LOCALID"))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass


def distributed_init(config):
    if config.distributed.world_size == 1:
        raise ValueError("Cannot initialize distributed with distributed_world_size=1")
    logger.info(f"XLA Mode:{is_xla()}")

    if is_xla():
        config.device_id = xm.get_local_ordinal()
        config.distributed.rank = xm.get_ordinal()
    elif dist.is_initialized():
        warnings.warn("Distributed is already initialized, cannot initialize twice!")
        config.distributed.rank = dist.get_rank()
    else:
        logger.info(
            f"Distributed Init (Rank {config.distributed.rank}): "
            f"{config.distributed.init_method}"
        )

        nccl_config = config.distributed.get("nccl", {})

        if nccl_config.get("nsocks_perthread", None):
            os.environ["NCCL_NSOCKS_PERTHREAD"] = str(nccl_config["nsocks_perthread"])
            logger.info(f"NCCL_NSOCKS_PERTHREAD: {os.environ['NCCL_NSOCKS_PERTHREAD']}")

        if nccl_config.get("socket_nthreads", None):
            os.environ["NCCL_SOCKET_NTHREADS"] = str(nccl_config["socket_nthreads"])
            logger.info(f"NCCL_SOCKET_NTHREADS: {os.environ['NCCL_SOCKET_NTHREADS']}")

        dist.init_process_group(
            backend=config.distributed.backend,
            init_method=config.distributed.init_method,
            world_size=config.distributed.world_size,
            rank=config.distributed.rank,
        )
        logger.info(
            f"Initialized Host {socket.gethostname()} as Rank "
            f"{config.distributed.rank}"
        )

        if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
            # Set for onboxdataloader support
            split = config.distributed.init_method.split("//")
            assert len(split) == 2, (
                "host url for distributed should be split by '//' "
                + "into exactly two elements"
            )

            split = split[1].split(":")
            assert (
                len(split) == 2
            ), "host url should be of the form <host_url>:<host_port>"
            os.environ["MASTER_ADDR"] = split[0]
            os.environ["MASTER_PORT"] = split[1]

        # perform a dummy all-reduce to initialize the NCCL communicator
        dummpy_tensor = torch.zeros(1)
        dist.all_reduce(dummpy_tensor.cuda())

        suppress_output(is_main())
        config.distributed.rank = dist.get_rank()
    return config.distributed.rank


def suppress_output(is_main):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    import warnings

    builtin_warn = warnings.warn

    def warn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main or force:
            builtin_warn(*args, **kwargs)

    # Log warnings only once
    warnings.warn = warn
    warnings.simplefilter("once", UserWarning)


def open_if_master(path, mode):
    from mmf.utils.file_io import PathManager

    if is_main():
        return PathManager.open(path, mode)
    else:
        return contextlib.nullcontext()


def open_if_main(*args):
    return open_if_master(*args)


def broadcast_xla_master_model_param(model):
    logger.info("Broadcasting XLA model parameters and buffers from master process ...")

    parameters_and_buffers = []
    for p in chain(model.parameters(), model.buffers()):
        # Set all params in non-master devices to zero so that all_reduce is equivalent
        # to broadcasting parameters from master to other devices.
        if not is_main():
            zero = torch.tensor(0, dtype=p.data.dtype, device=p.data.device)
            p.data.mul_(zero)
        parameters_and_buffers.append(p.data)
    xm.wait_device_ops()
    xm.all_reduce(xm.REDUCE_SUM, parameters_and_buffers)
    xm.mark_step()
    xm.rendezvous("mmf.trainers.core.device.broadcast_xla_master_model_param")
    logger.info("Done!")
