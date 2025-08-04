import pickle
from typing import Callable, Iterable

import torch
from torch import distributed as dist
from torch.utils.data import Dataset, Sampler
from torch.distributed.checkpoint.stateful import Stateful
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torchdata.stateful_dataloader import StatefulDataLoader

from halite.logging import logger


def _all_reduce(tensor, reduce_op, group):
    return funcol.all_reduce(tensor, reduce_op, group).item()


class DataManager:
    def __init__(self, loaders, process_group, check_finished=True):
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]

        self.loaders = loaders
        self.pg = process_group
        self.check_finished = check_finished
        self.finished = torch.tensor(0, dtype=torch.float32, device="cuda")

    def __iter__(self):
        self.loader_iter = iter(self.loaders[0])
        self.loader_idx = 0

        return self

    def __next__(self):
        finished = False

        try:
            batch = next(self.loader_iter)

        except StopIteration:
            self.loader_idx += 1

            if self.loader_idx < len(self.loaders):
                self.loader_iter = iter(self.loaders[self.loader_idx])

                try:
                    batch = next(self.loader_iter)

                except StopIteration:
                    finished = True

            else:
                finished = True

        if not self.check_finished:
            return batch

        # self.finished = self.finished.to("cuda")
        self.finished.fill_(float(finished))
        # dist.all_reduce(self.finished, group=self.pg, op=dist.ReduceOp.MAX)
        finished = _all_reduce(self.finished, c10d.ReduceOp.MAX.name, self.pg)
        # self.finished = self.finished.cpu()
        finished = finished > 0

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

    def state_dict(self):
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict):
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(f"No state for {self._rank_id} found in state_dict")

            return

        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))
