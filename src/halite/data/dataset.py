import bisect
import dataclasses
import os
import math
import itertools

try:
    import array_record.python.array_record_data_source as array_record
except ImportError:
    array_record = None

try:
    from ffrecord import FileReader

except ImportError:
    FileReader = None

import torch
from torch.utils import data

from halite.data.record import Record
from halite.data.index_shuffle import index_shuffle


@dataclasses.dataclass
class FileInstruction:
    filename: str
    skip: int
    take: int
    examples_in_shard: int


class FFRecordDataSource:
    def __init__(self, file_instructions):
        self.readers = [
            FileReader(instruction.filename, check_data=False)
            for instruction in file_instructions
        ]
        indices = [instruction.take for instruction in file_instructions]
        self.cumsum = list(itertools.accumulate(indices))
        self.length = sum(indices)
        self.skips = [instruction.skip for instruction in file_instructions]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        reader_id = bisect.bisect_right(self.cumsum, index)

        sample_id = index
        if reader_id > 0:
            sample_id = index - self.cumsum[reader_id - 1]

        if reader_id == len(self.skips):
            print(self.length, self.cumsum, self.skips, reader_id, sample_id)

        sample_id += self.skips[reader_id]

        return self.readers[reader_id].read_one(sample_id).tobytes()

    def __del__(self):
        for reader in self.readers:
            reader.close()


def build_dataset_from_spec(spec, split="train", split_ratio=0):
    datasources = {}

    for dataset, shards in spec.shards.items():
        instructions = []

        for shard_path, shard_size in shards.items():
            skip_size = 0

            if split == "train":
                if split_ratio != 0:
                    skip_size = max(math.floor(shard_size * split_ratio), 1)
                    skip_size = shard_size - skip_size

                instructions.append(
                    FileInstruction(
                        os.path.join(spec.root, dataset, shard_path),
                        skip_size,
                        shard_size - skip_size,
                        shard_size,
                    )
                )

            else:
                if split_ratio != 0:
                    skip_size = max(math.ceil(shard_size * split_ratio), 1)

                instructions.append(
                    FileInstruction(
                        os.path.join(spec.root, dataset, shard_path),
                        0,
                        skip_size,
                        shard_size,
                    )
                )

        if len(instructions) == 0:
            continue

        if instructions[0].filename.endswith(".ffr"):
            datasources[dataset] = FFRecordDataSource(instructions)

        elif instructions[0].filename.endswith(".arrayrecord"):
            datasources[dataset] = array_record.ArrayRecordDataSource(instructions)

        else:
            raise ValueError(f"Unsupported file extension: {instructions[0].filename}")

    datasets = []
    ratios = []
    names = []

    for name, dataset in datasources.items():
        names.append(name)
        ratios.append(spec.ratios[name])
        datasets.append(dataset)

    return datasets, ratios, names


class MapDataset(data.Dataset):
    def __init__(self, datasets, names=None, operations=None):
        self.datasets = datasets
        self.names = names
        self.points = torch.cumsum(torch.tensor([len(d) for d in datasets]), 0).tolist()
        self.operations = [] if operations is None else operations

    def __len__(self):
        return self.points[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.points, idx)
        dataset = self.datasets[dataset_idx]

        if dataset_idx == 0:
            sample_idx = idx

        else:
            sample_idx = idx - self.points[dataset_idx - 1]

        record = Record(
            data=dataset[sample_idx % len(dataset)],
            _meta_={
                "dataset_id": dataset_idx,
                "sample_id": sample_idx,
                "dataset_name": self.names[dataset_idx]
                if self.names is not None
                else None,
            },
        )

        record = [record]

        for op in self.operations:
            record = op(record)

        return next(iter(record))


def HFDataset(path, name=None, split="test", operations=None):
    from datasets import load_dataset

    return MapDataset(
        [load_dataset(path, name, split=split, trust_remote_code=True)],
        operations=operations,
    )


class WeightedIterableDataset(data.IterableDataset):
    def __init__(
        self,
        datasets,
        ratios,
        names=None,
        operations=None,
        num_replicas=1,
        rank=0,
        upsample=False,
        shuffle=True,
        seed=42,
        ensure_divisible_to=1,
    ):
        self.num_replicas = num_replicas
        self.rank = rank

        self.names = names
        self.ratios = ratios
        sizes = torch.tensor([len(d) for d in datasets])
        ratios = torch.as_tensor(ratios)
        ratios /= ratios.sum()
        n_sample = sizes.sum() * ratios

        if upsample:
            target_sample = n_sample / torch.min(n_sample / sizes)

        else:
            target_sample = n_sample / torch.max(n_sample / sizes)

        self.datasets = datasets
        self.target_sample = torch.round(target_sample).to(torch.int64)

        if ensure_divisible_to > 1:
            self.target_sample = (
                self.target_sample // ensure_divisible_to * ensure_divisible_to
            )

        self.points = torch.cumsum(self.target_sample, 0).tolist()
        self.operations = [] if operations is None else operations
        self.shuffle = shuffle
        self.seed = seed

        self._current_id = None
        self._sample_states = {}

    def __len__(self):
        return self.points[-1]

    def summary(self, total=True):
        res = []

        if self.names is not None:
            for i, (name, dset, ratio, sample) in enumerate(
                zip(self.names, self.datasets, self.ratios, self.target_sample.tolist())
            ):
                res.append(
                    f"#{i} {name} size: {len(dset)} ratio: {ratio} sample: {sample}"
                )

        else:
            for i, (dset, ratio, sample) in enumerate(
                zip(self.datasets, self.ratios, self.target_sample.tolist())
            ):
                res.append(f"#{i} size: {len(dset)} ratio: {ratio} sample: {sample}")

        if total:
            res.append(f"total: {len(self)}")

        return "\n".join(res)

    def __getitem__(self, idx):
        if self.shuffle:
            idx = index_shuffle(idx, len(self) - 1, self.seed, 4)

        dataset_idx = bisect.bisect_right(self.points, idx)
        dataset = self.datasets[dataset_idx]

        if dataset_idx == 0:
            sample_idx = idx

        else:
            sample_idx = idx - self.points[dataset_idx - 1]

        record = Record(
            data=dataset[sample_idx % len(dataset)],
            _meta_={
                "dataset_id": dataset_idx,
                "sample_id": sample_idx,
                "dataset_name": self.names[dataset_idx]
                if self.names is not None
                else None,
            },
        )

        return record

    def __iter__(self):
        worker_info = data.get_worker_info()

        self.set_worker_state(worker_info)

        return self.iterator()

    def set_worker_state(self, worker_info):
        if worker_info is None:
            offset = 0
            step = 1

        else:
            offset = worker_info.id
            step = worker_info.num_workers

        per_world = len(self) // self.num_replicas
        start = per_world * self.rank + offset
        end = start + per_world - offset

        self.worker_state = {
            "start": start,
            "end": end,
            "step": step,
            "worker_id": offset,
            "num_workers": step,
            "dp_rank": self.rank,
            "dp_size": self.num_replicas,
            "seed": self.seed,
        }

        if self._current_id is None:
            self._current_id = start

    def load_state_dict(self, state_dict):
        self._current_id = state_dict["current_id"]
        self._sample_states = state_dict["sample_states"]

        for op_id, op in enumerate(self.operations):
            if op_id in self._sample_states:
                op.load_state_dict(self._sample_states[op_id])

    def state_dict(self):
        return {
            "current_id": self._current_id,
            "sample_states": self._sample_states,
        }

    def _iterator(self):
        data = self.sample_iterator()

        for op in self.operations:
            data = op(data)

        return data

    def iterator(self):
        for sample in self._iterator():
            for op_id, op in enumerate(self.operations):
                if hasattr(op, "state_dict"):
                    self._sample_states[op_id] = op.state_dict()

            yield sample

    def sample_iterator(self):
        for idx in range(
            self._current_id,
            self.worker_state["end"],
            self.worker_state["step"],
        ):
            record = self[idx]

            self._current_id = idx + self.worker_state["step"]

            yield record
