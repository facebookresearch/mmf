# Copyright (c) Facebook, Inc. and its affiliates.

import os
import socket
import unittest

import torch
from mmf.common.registry import registry
from mmf.trainers.mmf_trainer import MMFTrainer
from mmf.utils.build import build_optimizer
from omegaconf import OmegaConf
from tests.test_utils import SimpleModel, skip_if_no_cuda
from tests.trainers.test_training_loop import TrainerTrainingLoopMock


try:
    from fairscale.optim.oss import OSS
    from fairscale.nn.data_parallel import ShardedDataParallel
    from fairscale.optim.grad_scaler import ShardedGradScaler

    FAIRSCALE_AVAILABLE = True
except ImportError:
    FAIRSCALE_AVAILABLE = False


GLOO_AVAILABLE = torch.distributed.is_gloo_available()
NCCL_AVAILABLE = torch.distributed.is_nccl_available()


def find_free_port() -> None:
    s = socket.socket()
    s.bind(("localhost", 0))  # Bind to a free port provided by the host.
    return s.getsockname()[1]


class MMFTrainerMock(TrainerTrainingLoopMock, MMFTrainer):
    def __init__(
        self,
        config,
        num_train_data,
        max_updates,
        max_epochs,
        device="cuda",
        fp16_model=True,
        num_workers=0,
    ):
        super().__init__(
            num_train_data,
            max_updates,
            max_epochs,
            fp16=fp16_model,
            num_workers=num_workers,
        )
        self.device = torch.device(device)
        self.config = OmegaConf.merge(self.config, config)
        self.model = SimpleModel({"in_dim": 1})
        self.model.build()
        self.model = self.model.cuda()
        self.optimizer = build_optimizer(self.model, self.config)
        self.distributed = True
        self.local_rank = 0
        self.parallelize_model()
        self.load_fp16_scaler()


class TestShardedDDP(unittest.TestCase):
    def setUp(self):
        self.config_oss = OmegaConf.create(
            {
                "optimizer": {
                    "type": "adam_w",
                    "enable_state_sharding": True,
                    "params": {"lr": 5e-5},
                },
                "training": {"find_unused_parameters": False},
            }
        )
        self.config_no_oss = OmegaConf.create(
            {
                "optimizer": {
                    "type": "adam_w",
                    "enable_state_sharding": False,
                    "params": {"lr": 5e-5},
                },
                "training": {"find_unused_parameters": False},
            }
        )

    def tearDown(self):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        registry.unregister("distributed")

    def _init_distributed_env(self, backend: str = "gloo") -> None:
        """ Initialize a single process in distributed environment. """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        torch.distributed.init_process_group(backend, rank=0, world_size=1)

    def _test_no_sharding(self, backend="gloo"):
        self._init_distributed_env(backend)
        trainer = MMFTrainerMock(self.config_no_oss, 100, 2, 0.04)
        trainer.load_datasets()

        self.assertFalse(isinstance(trainer.optimizer, OSS))

        self.assertFalse(isinstance(trainer.model, ShardedDataParallel))
        self.assertTrue(
            isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel)
        )

        self.assertFalse(isinstance(trainer.scaler, ShardedGradScaler))
        self.assertTrue(isinstance(trainer.scaler, torch.cuda.amp.GradScaler))
        self.assertEqual(trainer.current_iteration, 0)
        trainer.training_loop()
        self.assertEqual(trainer.current_iteration, 4)
        del trainer

    def _test_sharding(self, backend="gloo"):
        self._init_distributed_env(backend)
        trainer = MMFTrainerMock(self.config_oss, 100, 2, 0.04)
        trainer.load_datasets()

        self.assertTrue(isinstance(trainer.optimizer, OSS))

        self.assertTrue(isinstance(trainer.model, ShardedDataParallel))

        self.assertTrue(isinstance(trainer.scaler, ShardedGradScaler))

        self.assertEqual(trainer.current_iteration, 0)
        trainer.training_loop()
        self.assertEqual(trainer.current_iteration, 4)
        del trainer

    @unittest.skip
    @skip_if_no_cuda
    @unittest.skipUnless(FAIRSCALE_AVAILABLE, "Tests for fairscale")
    @unittest.skipUnless(GLOO_AVAILABLE, "Tests for gloo backend")
    def test_sharding_gloo(self):
        self._test_sharding(backend="gloo")

    @unittest.skip
    @skip_if_no_cuda
    @unittest.skipUnless(FAIRSCALE_AVAILABLE, "Tests for fairscale")
    @unittest.skipUnless(GLOO_AVAILABLE, "Tests for gloo backend")
    def test_no_sharding_gloo(self):
        self._test_no_sharding(backend="gloo")

    @unittest.skip
    @skip_if_no_cuda
    @unittest.skipUnless(FAIRSCALE_AVAILABLE, "Tests for fairscale")
    @unittest.skipUnless(NCCL_AVAILABLE, "Tests for NCCL backend")
    def test_sharding_nccl(self):
        self._test_sharding(backend="nccl")

    @unittest.skip
    @skip_if_no_cuda
    @unittest.skipUnless(FAIRSCALE_AVAILABLE, "Tests for fairscale")
    @unittest.skipUnless(NCCL_AVAILABLE, "Tests for NCCL backend")
    def test_no_sharding_nccl(self):
        self._test_no_sharding(backend="nccl")
