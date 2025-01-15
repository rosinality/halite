import argparse

import torch

from halite.data.dataloader import DataLoader
from halite.data.dataset import WeightedIterableDataset
from halite.data import preprocess


class TestDataset:
    def __init__(self, n_samples, max_length, index_multiplier=1000):
        self.n_samples = n_samples
        self.max_length = max_length
        self.index_multiplier = index_multiplier

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.index_multiplier

        return list(range(start, start + idx % self.max_length + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--break_at", type=int, default=3)

    args = parser.parse_args()

    preprocess_ops = [
        preprocess.SequencePacking(args.length, key="data"),
    ]
    ds = WeightedIterableDataset(
        [TestDataset(args.n_samples, args.length)],
        [1.0],
        ["test"],
        num_replicas=1,
        rank=0,
        operations=preprocess_ops,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        num_workers=args.n_workers,
        collate_fn=preprocess.Collator(keys=("data",)),
        rank=0,
        check_finished=False,
        shuffle=False,
    )

    ref_data = []
    ref_ids = set()
    for batch_id, batch in enumerate(loader):
        ref_data.append(batch["data"])
        ref_ids.update(batch["data"].reshape(-1).tolist())

    data_ids = set()
    for i in range(len(ds)):
        data_ids.update(ds[i]["data"])

    # print(sorted(list(data_ids - ref_ids)))

    compare_data = []
    for batch_id, batch in enumerate(loader):
        compare_data.append(batch["data"])

        if batch_id == args.break_at:
            state_dict = loader.state_dict()
            break

    loader.load_state_dict(state_dict)

    for batch_id, batch in enumerate(loader):
        compare_data.append(batch["data"])

    for i, (compare, ref) in enumerate(zip(compare_data, ref_data)):
        print(i)
        print(compare)
        print(ref)

    import pickle

    print(pickle.loads(state_dict["dp_rank_0"]))

    for i, (compare, ref) in enumerate(zip(compare_data, ref_data)):
        assert torch.all(compare == ref), f"{compare} != {ref} at {i}"
