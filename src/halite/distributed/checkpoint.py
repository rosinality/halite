import functools
import enum
import time
from typing import Any
import os
import shutil
import re
from multiprocessing import get_context

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful

from halite.logging import logger
from halite.utils import get_torch_dtype
from halite.distributed.managers import (
    ModelManager,
    OptimizerManager,
    LRSchedulerManager,
)


class IntervalType(enum.Enum):
    STEPS = enum.auto()
    SECONDS = enum.auto()


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


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
        model_parts: list[nn.Module],
        optimizers: list[torch.optim.Optimizer],
        lr_schedulers: list[torch.optim.lr_scheduler.LRScheduler],
        dataloader: DataLoader,
        model_config: Any | None = None,
        states: dict[str, Any] = None,
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

        if not self.enable_checkpoint:
            return

        self.states = {} if states is None else states

        model_manager = (
            ModelManager(model_parts)
            if not isinstance(model_parts, ModelManager)
            else model_parts
        )
        optimizer_manager = (
            OptimizerManager(model_parts, optimizers)
            if not isinstance(optimizers, OptimizerManager)
            else optimizers
        )
        lr_scheduler_manager = (
            LRSchedulerManager(optimizers, lr_schedulers)
            if not isinstance(lr_schedulers, LRSchedulerManager)
            else lr_schedulers
        )

        self.states.update(
            {
                "model": model_manager,
                "optimizer": optimizer_manager,
                "dataloader": dataloader,
            }
        )
        self.states.update(lr_scheduler_manager.get_state_dict())

        self.model_config = model_config

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

    def _should_save(self, current_step: int, force: bool = False):
        if not self.enable_checkpoint:
            return False

        if not force:
            if (
                self.interval_type == IntervalType.STEPS
                and not current_step % self.interval == 0
            ):
                return False

            if self.interval_type == IntervalType.SECONDS:
                time_sync_result = (time.monotonic() - self.begin_time) >= self.interval
                self.time_sync_result = torch.tensor(int(time_sync_result))

                if self.time_sync_work is None:
                    self.time_sync_work = dist.all_reduce(
                        self.time_sync_result, group=self.pg, async_op=True
                    )

                    return False

                elif current_step % 5 == 4:
                    self.time_sync_work.wait()
                    self.time_sync_work = None
                    time_sync_result = self.time_sync_result.item()
                    self.time_sync_result = None

                    if time_sync_result == 0:
                        return False

                else:
                    return False

        if self.time_sync_work:
            self.time_sync_work.wait()
            self.time_sync_work = None
            self.time_sync_result = None

        return True

    def _async_wait(self):
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            logger.debug(
                f"Waiting for the checkpoint background process to finish, {time.monotonic()=}.:.2f"
            )

            if not self.mp.is_alive():
                raise RuntimeError("Checkpoint background process is not alive.")

            _ = self.mp_queue_recv.get()

        elif self.async_mode == AsyncMode.ASYNC:
            if self.async_future is not None:
                self.async_future.result()

    def _async_with_pinned_memory(self, checkpoint_id: str):
        try:
            from torch.distributed._state_dict_utils import (
                _copy_state_dict,
                _create_cpu_state_dict,
            )

        except ImportError as e:
            raise ImportError(
                "Please install the latest pytorch to use async checkpointing with pinned memory."
            )

        state_dict = dcp.state_dict_saver._stateful_to_state_dict(self.states)

        if self.cpu_offload_state_dict is None:
            logger.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(
                state_dict, pin_memory=True, share_memory=True
            )

        logger.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict, self.cpu_offload_state_dict, non_blocking=True
            )
            self.staging = True
            self.staging_id = checkpoint_id

    def save(self, current_step: int, force: bool = False):
        if not self._should_save(current_step, force):
            return

        begin = time.monotonic()
        checkpoint_id = self._create_checkpoint_id(current_step)
        self._async_wait()

        if self.model_config is not None:
            self.model_config.save(checkpoint_id)

        if force:
            self._save_last_step(current_step)

        elif self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._async_with_pinned_memory(checkpoint_id)

        elif self.async_mode == AsyncMode.ASYNC:
            self.async_future = dcp.async_save(
                self.states, checkpoint_id=checkpoint_id, process_group=self.pg
            )

        else:
            dcp.save(self.states, checkpoint_id=checkpoint_id)

        self.reset()
        self._purge_stale_checkpoints()

        logger.info(
            "Finished saving the checkpoint (or staging if async is enabled) in "
            f"{time.monotonic() - begin:.2f} seconds."
        )

    def maybe_wait_for_staging(self):
        if (
            self.enable_checkpoint
            and self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
            and self.staging
        ):
            if not self.staging_stream.query():
                self.staging_stream.synchronize()

            def sync_func():
                self.mp_queue_send.put_nowait(
                    (self.cpu_offload_state_dict, self.staging_id)
                )

            sync_func()
            self.staging = False

    def load(self, step: int = -1):
        if not self.enable_checkpoint:
            return False

        if not os.path.isdir(self.directory):
            return False

        if step != -1 and not os.path.isdir(self._create_checkpoint_id(step)):
            return False

        if step == -1:
            step_counts = []

            for filename in os.listdir(self.directory):
                match = re.search(r"step-(\d+)", filename)
                metadata_probe = os.path.join(self.directory, filename, ".metadata")

                if match and os.path.isfile(metadata_probe):
                    step_counts.append(int(match.group(1)))

            if not step_counts:
                return False

            step = max(step_counts)

        states = {"model": self.states["model"]} if step == 0 else self.states
        original_stateful_states = {
            k: v for k, v in states.items() if isinstance(v, Stateful)
        }
        logger.info(f"Loading the checkpoint at step {step}")
        begin = time.monotonic()
        dcp.load(states, checkpoint_id=self._create_checkpoint_id(step))
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds"
        )

        states.update(original_stateful_states)

        return True

    def _purge_stale_checkpoints(self):
        if self.keep_latest_k <= 0:
            return

        discovered_checkpoints = []
        for filename in os.listdir(self.directory):
            match = re.search(r"step-(\d+)", filename)
            path = os.path.join(self.directory, filename)
            discovered_checkpoints.append((int(match.group(1)), path))

        discovered_checkpoints.sort()
        to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

        for _, path in to_delete:
            logger.info(f"Deleting old checkpoint {path}")
            shutil.rmtree(path, ignore_errors=True)


def load_checkpoint(
    path: str, model_parts=None, optimizers=None, states=None, verbose: bool = False
):
    if states is None:
        states = {}

    if model_parts is not None:
        states["model"] = ModelManager(model_parts)

    if optimizers is not None:
        states["optimizer"] = OptimizerManager(model_parts, optimizers)

    original_stateful_states = {
        k: v for k, v in states.items() if isinstance(v, Stateful)
    }

    if verbose:
        logger.info(f"Loading the checkpoint of {path}")
        begin = time.monotonic()

    dcp.load(states, checkpoint_id=path)
    states.update(original_stateful_states)

    if verbose:
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds"
        )
