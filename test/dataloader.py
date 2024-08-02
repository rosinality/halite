import argparse
import json
from collections import Counter
from random import sample

from torch.utils.data import DataLoader
from tqdm import tqdm

from halite.data.tokenizers.llama3 import Llama3Tokenizer
from halite.data.dataset import build_dataset_from_spec, WeightedIterableDataset
from halite.data import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("spec", type=str)

    args = parser.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)

    tokenizer = Llama3Tokenizer(args.tokenizer)

    datasets, ratios, names = build_dataset_from_spec(spec)
    preprocess_ops = [
        preprocess.ParseFeatures(),
        preprocess.Tokenize(tokenizer, bos=True, eos=True),
        preprocess.SequencePacking(args.length),
    ]

    dset_counter = Counter()
    sample_counter = {"A": set(), "B": set()}

    for rank in range(2):
        ds = WeightedIterableDataset(
            datasets,
            ratios,
            names,
            num_replicas=2,
            rank=rank,
            operations=preprocess_ops,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch,
            num_workers=args.n_workers,
            collate_fn=preprocess.Collator(keys=("text",)),
        )

        pbar = tqdm(loader)
        for batch in pbar:
            for seq in batch.text.tolist():
                subseqs = []
                current_seq = []

                for elem in seq:
                    if elem == tokenizer.bos_id:
                        continue

                    if elem == tokenizer.eos_id:
                        subseqs.append(current_seq)
                        current_seq = []

                        continue

                    current_seq.append(elem)

                if len(current_seq) != 0:
                    subseqs.append(current_seq)

                for seq in subseqs:
                    seq = tokenizer.decode(seq).strip()

                    seq = [int(elem) for elem in seq.split()]

                    if len(seq) == 0:
                        continue

                    if seq[0] < 1000:
                        dset_counter["A"] += 1
                        sample_counter["A"].add(seq[0])

                    else:
                        dset_counter["B"] += 1
                        sample_counter["B"].add(seq[0])

            total = dset_counter["A"] + dset_counter["B"]
            pbar.set_description(
                f"A: {round(dset_counter['A'] / total, 2)} B: {round(dset_counter['B'] / total, 2)}"
            )

        print(batch._meta_)

    for i in range(1000):
        if i not in sample_counter["A"]:
            print("A", i)
