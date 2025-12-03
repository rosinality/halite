import argparse
from glob import glob
import os
import json

try:
    import array_record.python.array_record_data_source as array_record

except ImportError:
    array_record = None

try:
    from ffrecord import FileReader

except ImportError:
    FileReader = None

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--ext", type=str, default="arrayrecord")
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    files = sorted(glob(os.path.join(args.path, f"**/*.{args.ext}"), recursive=True))
    dataset_collection = {}

    rel_root = args.path

    for file in tqdm(files):
        dirname = os.path.dirname(file)

        if args.depth != 0:
            rel_dir = os.path.relpath(dirname, args.path)
            rel_root = os.sep.join(
                os.path.normpath(rel_dir).split(os.sep)[: args.depth]
            )
            dataset = rel_root
            dirname = os.path.join(args.path, dataset)

        else:
            dataset = os.path.relpath(dirname, rel_root)

        rel_file = os.path.relpath(file, dirname)

        if file.endswith(".arrayrecord"):
            ds = array_record.ArrayRecordDataSource(file)
            length = len(ds)

        elif file.endswith(".ffr"):
            ds = FileReader(file, check_data=False)
            length = ds.n
            ds.close()

        else:
            raise ValueError(f"Unsupported file extension: {file}")

        if dataset not in dataset_collection:
            dataset_collection[dataset] = {rel_file: length}

        else:
            dataset_collection[dataset][rel_file] = length

    total_sizes = {}

    for dataset, shards in dataset_collection.items():
        total_sizes[dataset] = sum(shards.values())

    ratios = {}
    total_count = sum(total_sizes.values())
    for dataset, size in total_sizes.items():
        ratios[dataset] = size / total_count

    config = {"root": args.path, "ratios": ratios, "shards": dataset_collection}
    serialized = json.dumps(config, indent=4)

    print(serialized)

    if args.output is not None:
        with open(args.output, "w") as f:
            f.write(serialized)
