import pickle
from typing import Callable, Iterable

import torch
from torch import distributed as dist
from torch.utils.data import Dataset, Sampler
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

from halite.logging import logger


class DataManager:
    def __init__(self, loader, mesh, device="cuda"):
        self.loader = loader
        self.mesh = mesh
        self.finished = torch.tensor(0, dtype=torch.float32, device=device)

    def __iter__(self):
        self.loader_iter = iter(self.loader)

        return self

    def __next__(self):
        try:
            batch = next(self.loader_iter)

        except StopIteration:
            finished = True

        else:
            finished = False

        self.finished.fill_(float(finished))
        dist.all_reduce(
            self.finished, group=self.mesh.get_group("dp"), op=dist.ReduceOp.MAX
        )
        finished = self.finished.item() > 0

        if finished:
            raise StopIteration

        return batch


class DataLoader(StatefulDataLoader, Stateful):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[list] | Iterable[list] | None = None,
        rank: int = 0,
        mesh: DeviceMesh | None = None,
        check_finished: bool = True,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        snapshot_every_n_steps: int | None = 1,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            snapshot_every_n_steps=snapshot_every_n_steps,
        )

        self._rank = rank
        self._rank_id = f"dp_rank_{rank}"
        self._mesh = mesh
        self._check_finished = check_finished

    def state_dict(self):
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict):
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(f"No state for {self._rank_id} found in state_dict")

            return

        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))

    def __next__(self):
        try:
            batch = next(self.loader_iter)

        except StopIteration:
            finished = True

        else:
            finished = False

        if not self._check_finished:
            return batch

        self.finished.fill_(float(finished))
        dist.all_reduce(
            self.finished, group=self._mesh.get_group("dp"), op=dist.ReduceOp.MAX
        )
        finished = self.finished.item() > 0

        if finished:
            raise StopIteration

        return batch
