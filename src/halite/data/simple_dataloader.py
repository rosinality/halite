import copy
import itertools
import random
from typing import Any
from multiprocessing import Process, Queue, Event
from queue import Full, Empty


import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data._utils import HAS_NUMPY
from torch.utils.data._utils.worker import _generate_state, WorkerInfo


def feed_buffer(
    queue, stop_event, iterator, worker_id, n_workers, base_seed, state_dict
):
    seed = base_seed + worker_id

    torch.utils.data._utils.worker._worker_info = WorkerInfo(
        id=worker_id, num_workers=n_workers, seed=seed, dataset=iterator
    )

    random.seed(seed)
    torch.manual_seed(seed)

    if HAS_NUMPY:
        np_seed = _generate_state(seed, worker_id)
        import numpy as np

        np.random.seed(np_seed)

    return_state_dict = hasattr(iterator, "state_dict")

    if hasattr(iterator, "load_state_dict") and state_dict is not None:
        iterator.load_state_dict(state_dict)

    for item in iterator:
        while not stop_event.is_set():
            state_dict = None
            if return_state_dict:
                state_dict = copy.deepcopy(iterator.state_dict())

            try:
                queue.put((item, False, state_dict), timeout=0.1)

                break

            except Full:
                pass

        if stop_event.is_set():
            break

    queue.put((None, True, None))
    stop_event.wait()


def consume_buffer(producers, queues, yield_worker_id, finished_workers):
    if finished_workers is None:
        finished_workers = set()

    worker_ids = list(range(len(queues)))
    cycle = itertools.cycle(worker_ids[yield_worker_id:] + worker_ids[:yield_worker_id])
    worker_id = next(cycle)

    while True:
        try:
            if worker_id in finished_workers:
                worker_id = next(cycle)

            queue = queues[worker_id]
            item, finished, state_dict = queue.get(timeout=0.1)

            if item is not None and not finished:
                yield item, worker_id, finished, state_dict

            if finished:
                finished_workers.add(worker_id)

            worker_id = next(cycle)

        except Empty:
            pass

        if len(finished_workers) == len(producers):
            break


def async_iterator(
    iterator, buffer_size: int, n_workers: int, state_dict: dict[int, Any]
):
    queues = [Queue(buffer_size) for _ in range(n_workers)]
    stop_event = Event()
    producers = []

    base_seed = state_dict["_base_seed"]
    yield_worker_id = state_dict.get("_yield_worker_id", 0)
    finished_workers = state_dict.get("_finished_workers", set())

    for i in range(n_workers):
        state = state_dict["_worker_snapshots"].get(i, None)

        producer = Process(
            target=feed_buffer,
            args=(
                queues[i],
                stop_event,
                iterator,
                i,
                n_workers,
                base_seed,
                state,
            ),
        )
        producer.start()
        producers.append(producer)

    consumer = consume_buffer(producers, queues, yield_worker_id, finished_workers)

    try:
        yield from consumer

    finally:
        stop_event.set()
        consumer.close()

        for producer in producers:
            producer.join(timeout=0.2)

            if producer.exitcode is None:
                producer.kill()


class BatchIterator:
    def __init__(self, iterator, batch_size=1, collate_fn=None):
        self.iterator = iterator
        self.batch_size = batch_size

        self.collate_fn = collate_fn
        if collate_fn is None:
            self.collate_fn = default_collate

        self._state_dict = {}

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

        if hasattr(self.iterator, "load_state_dict"):
            self.iterator.load_state_dict(state_dict)

    def __iter__(self):
        self._iterator = iter(self.iterator)

        return self

    def __next__(self):
        batch = []

        while True:
            try:
                item = next(self._iterator)

            except StopIteration as e:
                raise e

            batch.append(item)

            if len(batch) == self.batch_size:
                if hasattr(self.iterator, "state_dict"):
                    self._state_dict = self.iterator.state_dict()

                return self.collate_fn(batch)


class SimpleDataLoader:
    def __init__(self, dataset, rank=0, prefetch_factor=2, num_workers=1):
        self.dataset = dataset
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers

        self._iterator = None
        self._rank = rank
        self._rank_id = f"dp_rank_{rank}"

        self._init_state_dict()

    def _init_state_dict(self):
        self._state_dict = {}
        self._state_dict["_yield_worker_id"] = 0
        self._state_dict["_base_seed"] = (
            torch.empty((), dtype=torch.int64).random_().item()
        )
        self._state_dict["_worker_snapshots"] = {}
        self._state_dict["_finished_workers"] = set()
        self._finished = False

    def state_dict(self):
        return {self._rank_id: self._state_dict}

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict[self._rank_id]

    def __iter__(self):
        if self._finished:
            self._init_state_dict()

        self._iterator = async_iterator(
            self.dataset, self.prefetch_factor, self.num_workers, self._state_dict
        )

        return self

    def __next__(self):
        try:
            item, worker_id, finished, state_dict = next(self._iterator)

        except StopIteration as e:
            self._finished = True

            raise e

        self._state_dict["_worker_snapshots"][worker_id] = state_dict
        self._state_dict["_yield_worker_id"] = worker_id + 1

        if finished:
            self._state_dict["_finished_workers"].add(worker_id)

        return item
