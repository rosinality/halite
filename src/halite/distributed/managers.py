from collections.abc import Sequence
import functools
from typing import Any

from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful


class ModelManager(Stateful):
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


class OptimizerManager(Stateful):
    def __init__(self, model_parts, optimizers=None, optimizer_fn=None):
        self.model_parts = (
            [model_parts] if not isinstance(model_parts, Sequence) else model_parts
        )

        self.optimizers = optimizers
        if self.optimizers is not None and not isinstance(self.optimizers, Sequence):
            self.optimizers = [self.optimizers]

        if self.optimizers is None:
            self.optimizers = [
                optimizer_fn(part.parameters()) for part in self.model_parts
            ]

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def state_dict(self):
        fn = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

        return {
            k: v
            for sd in map(fn, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict):
        fn = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

        list(map(fn, self.model_parts, self.optimizers))


class LRSchedulerManager:
    def __init__(self, optimizers, schedulers=None, scheduler_fn=None):
        if isinstance(optimizers, OptimizerManager):
            optimizers = optimizers.optimizers

        if not isinstance(optimizers, Sequence):
            optimizers = [optimizers]

        self.optimizers = optimizers
        self.schedulers = schedulers

        if self.schedulers is not None and not isinstance(self.schedulers, Sequence):
            self.schedulers = [self.schedulers]

        if self.schedulers is None:
            self.schedulers = [scheduler_fn(optimizer) for optimizer in self.optimizers]

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_state_dict(self):
        state_dict = {}

        if len(self.schedulers) == 1:
            state_dict["lr_scheduler"] = self.schedulers[0]

        else:
            for i, scheduler in enumerate(self.schedulers):
                state_dict[f"lr_scheduler_{i}"] = scheduler

        return state_dict
