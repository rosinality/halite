import argparse

import torch

from halite.data.record import Record
from halite.data.dataloader import DataLoader
from halite.data.dataset import WeightedIterableDataset
from halite.data import preprocess
from halite.projects.sft.preprocess import SFTSequencePacking


class TestDataset:
    def __init__(self, n_samples, max_length, index_multiplier=1000):
        self.n_samples = n_samples
        self.max_length = max_length
        self.index_multiplier = index_multiplier

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.index_multiplier
        tokens = list(range(start, start + idx % self.max_length + 1))

        return tokens


class SFTSample:
    def __call__(self, iterator):
        for record in iterator:
            record.input = torch.tensor(record.data)
            record.target = torch.tensor(record.data)

            yield record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--break_at", type=int, default=3)

    args = parser.parse_args()

    preprocess_ops = [
        SFTSample(),
        SFTSequencePacking(args.length),
    ]
    ds = WeightedIterableDataset(
        [TestDataset(args.n_samples, args.length)],
        [1.0],
        ["test"],
        num_replicas=1,
        rank=0,
        operations=preprocess_ops,
        shuffle=False,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        num_workers=args.n_workers,
        collate_fn=preprocess.Collator(keys=("input", "target")),
        rank=0,
        shuffle=False,
    )

    ref_data = []
    ref_ids = set()
    for batch_id, batch in enumerate(loader):
        ref_data.append(batch["input"])
        ref_ids.update(batch["input"].reshape(-1).tolist())

    # print(sorted(list(data_ids - ref_ids)))

    compare_data = []
    for batch_id, batch in enumerate(loader):
        compare_data.append(batch["input"])

        if batch_id == args.break_at:
            state_dict = loader.state_dict()
            break

    loader.load_state_dict(state_dict)

    print("RESUME")

    import pickle

    print(pickle.loads(state_dict["dp_rank_0"]))

    for batch_id, batch in enumerate(loader):
        compare_data.append(batch["input"])

    for i, (compare, ref) in enumerate(zip(compare_data, ref_data)):
        print(i)
        print(ref)
        print(compare)

    for i, (compare, ref) in enumerate(zip(compare_data, ref_data)):
        assert torch.all(compare == ref), f"{compare} != {ref} at {i}"
