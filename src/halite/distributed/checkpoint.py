import functools
from dataclasses import dataclass, field
import enum
from io import BytesIO
import time
from typing import Any
import os
from multiprocessing import get_context

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

from halite.logging import logger
from halite.utils import get_torch_dtype


class IntervalType(enum.Enum):
    STEPS = enum.auto()
    SECONDS = enum.auto()


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model

    def state_dict(self) -> dict[str, Any]:
        return {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class OptimizerWrapper(Stateful):
    def __init__(
        self,
        model: nn.Module | list[nn.Module],
        optim: torch.optim.Optimizer | list[torch.optim.Optimizer],
    ) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.optim = [optim] if isinstance(optim, torch.optim.Optimizer) else optim

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for sd in map(func, self.model, self.optim) for k, v in sd.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model, self.optim))


class Terminate:
    pass


class SaveDone:
    pass


def checkpoint_worker(recv, send):
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group()
    try:
        while True:
            logger.debug("Checkpoint background process is done.")
            send.put(SaveDone())
            logger.debug("Wait for the new state_dict.")
            obj = recv.get()
            logger.debug("Received the new state_dict.")
            if isinstance(obj, Terminate):
                logger.info("Terminating the checkpoint background process.")
                return
            assert isinstance(obj, tuple)
            begin = time.monotonic()
            state, checkpoint_id = obj
            dcp.save(state, checkpoint_id=checkpoint_id)
            logger.info(
                "Finish saving the checkpoint in the background process in "
                f"{time.monotonic() - begin:.2f} seconds."
            )
    finally:
        logger.info("Destroying the process group.")
        dist.destroy_process_group()


class CheckpointManager:
    def __init__(
        self,
        dataloader: DataLoader,
        model_parts: list[nn.Module],
        optimizers: list[torch.optim.Optimizer],
        lr_schedulers: list[torch.optim.lr_scheduler.LRScheduler],
        states: dict[str, Any],
        directory: str = "checkpoint",
        interval_type: str = "steps",
        interval: int = 500,
        enable_checkpoint: bool = True,
        keep_latest_k: int = 0,
        model_weights_only: bool = False,
        export_dtype: str = "float32",
        async_mode: str = "disabled",
    ):
        self.enable_checkpoint = enable_checkpoint
        self.keep_latest_k = keep_latest_k

        assert len(model_parts) == len(optimizers)
        assert len(model_parts) == len(lr_schedulers)

        self.states = states
        self.states.update(
            {
                "model": ModelWrapper(model_parts),
                "optimizer": OptimizerWrapper(model_parts, optimizers),
                "dataloader": dataloader,
            }
        )

        if len(lr_schedulers) == 1:
            self.states["lr_scheduler"] = lr_schedulers[0]

        else:
            for i, lr_scheduler in enumerate(lr_schedulers):
                self.states[f"lr_scheduler_{i}"] = lr_scheduler

        self.directory = directory
        self.interval_type = (
            IntervalType.SECONDS if interval_type == "seconds" else IntervalType.STEPS
        )
        self.interval = interval
        self.begin_time = 0
        self.time_sync_work = None
        self.time_sync_result = None

        self.pg = dist.new_group(backend="gloo")

        self.model_weights_only = model_weights_only
        self.export_dtype = get_torch_dtype(export_dtype)

        self.mp = None
        async_mode = async_mode.lower()

        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED

        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
            self.async_future = None

        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM

            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=checkpoint_worker,
                args=(self.mp_queue_recv, self.mp_queue_send),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None
            self.staging = False
            self.staging_id = None
            self.staging_stream = torch.cuda.Stream()

        else:
            raise ValueError(f"Invalid async mode: {async_mode}")

    def __del__(self):
        if self.enable_checkpoint and self.mp and self.mp.is_alive():
            self.mp_queue_send.put(Terminate())
            self.mp.join()

    def reset(self):
        self.begin_time = time.monotonic()

    def _create_checkpoint_id(self, step: int):
        return os.path.join(self.directory, f"step-{step}")

    def _save_last_step(self, cur_step: int):
        if self.model_weights_only:
            self.states = self.states["model"].state_dict()

            if self.export_dtype != torch.float32:
                self.states = {
                    k: v.to(self.export_dtype) for k, v in self.states.items()
                }

        dcp.save(self.states, checkpoint_id=self._create_checkpoint_id(cur_step))
        self.reset()
