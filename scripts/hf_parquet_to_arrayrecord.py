import argparse
import random
import fnmatch

from tqdm import tqdm
from array_record.python.array_record_module import ArrayRecordWriter

import os
import tempfile
import math
import urllib
import multiprocessing as mp
from functools import partial

import fsspec


def write_arrayrecord(parquet_files, output_fp, protocol, pbar=None, batch_size=1000):
    import pyarrow.parquet as pq

    fs = fsspec.filesystem(protocol)

    for file in parquet_files:
        if pbar is not None:
            basename = os.path.basename(file)
            pbar.set_description(basename)

        with fs.open(file, "rb") as f:
            with pq.ParquetFile(f) as pqf:
                for batch in pqf.iter_batches(batch_size=batch_size):
                    documents = batch.to_pylist()

                    for document in documents:
                        output_fp.write(document["text"].encode("utf-8"))

                        if pbar is not None:
                            pbar.update(1)


def parquet_to_arrayrecord(
    index_filenames, protocol, output_path, prefix, max_file, batch_size=1000
):
    index, filenames = index_filenames
    pbar = tqdm(dynamic_ncols=True, position=index % 16)
    digits = math.ceil(math.log10(max_file))
    shard_id = str(index + 1).zfill(digits)
    max_shard_id = str(max_file).zfill(digits)
    basename = f"{prefix}-{shard_id}-of-{max_shard_id}.arrayrecord"

    writer = ArrayRecordWriter(os.path.join(output_path, basename), "group_size:1")
    write_arrayrecord(filenames, writer, protocol, pbar, batch_size)
    writer.close()


def parquet_to_arrayrecord_gcp(
    index_filenames, protocol, output_path, prefix, max_file, batch_size=1000
):
    index, filenames = index_filenames

    from google.cloud import storage

    pbar = tqdm(dynamic_ncols=True, position=index % 16)

    with tempfile.NamedTemporaryFile() as tmp:
        writer = ArrayRecordWriter(tmp.name, "group_size:1")
        write_arrayrecord(filenames, writer, protocol, pbar, batch_size)
        writer.close()

        digits = math.ceil(math.log10(max_file))
        shard_id = str(index + 1).zfill(digits)
        max_shard_id = str(max_file).zfill(digits)
        basename = f"{prefix}-{shard_id}-of-{max_shard_id}.arrayrecord"
        client = storage.Client()
        gcs_path = urllib.parse.urlparse(output_path)
        bucket = client.get_bucket(gcs_path.hostname)
        gcs_prefix = gcs_path.path.lstrip("/")
        blob = bucket.blob(os.path.join(gcs_prefix, basename))
        blob.upload_from_filename(tmp.name)


def parse_size(size_str):
    """
    Parse a human-readable size string and convert it to bytes.

    Supports formats like '10GB', '100MB', '1TB', etc.
    Case-insensitive.
    """
    size_str = size_str.strip().upper()
    if size_str[-2:] in ["KB", "MB", "GB", "TB"]:
        number = float(size_str[:-2])
        unit = size_str[-2:]
    elif size_str[-1:] in ["K", "M", "G", "T"]:
        number = float(size_str[:-1])
        unit = size_str[-1:] + "B"
    else:
        try:
            return int(size_str)
        except ValueError:
            raise ValueError(f"Invalid size format: {size_str}")

    units = {"KB": 1, "MB": 2, "GB": 3, "TB": 4}
    return int(number * (1024 ** units[unit]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    parser.add_argument("--gcp", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--pattern", type=str, default="*.parquet")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--shard_size", type=str, default=None)
    parser.add_argument("--max_size", type=str, default=None)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    fs = fsspec.filesystem("hf")

    files = fs.find(args.path)
    files = fnmatch.filter(files, args.pattern)

    if args.shuffle:
        random.shuffle(files)

    filenames = []

    total_size = 0
    target_size = None
    target_shard_size = None
    if args.max_size is not None:
        target_size = parse_size(args.max_size)
    if args.shard_size is not None:
        target_shard_size = parse_size(args.shard_size)

    pbar = tqdm(files, dynamic_ncols=True)
    current_shard = []
    current_shard_size = 0
    shard_id = 0
    for i, file in enumerate(pbar):
        size = fs.info(file)["size"]
        total_size += size
        pbar.set_description(f"{total_size / (1024 ** 3):.2f}GB")

        if target_size is not None and total_size > target_size:
            break

        if target_shard_size is not None:
            current_shard_size += size

            if current_shard_size > target_shard_size:
                filenames.append((shard_id, current_shard))
                shard_id += 1
                current_shard = []
                current_shard_size = size

            current_shard.append(file)

        else:
            filenames.append((shard_id, (file,)))

    print("total shards", len(filenames))
    print(
        "average raw file count per shard",
        sum([len(shard) for _, shard in filenames]) / len(filenames),
    )
    print("first 3 shards")
    for i, (index, files) in enumerate(filenames[:3]):
        print("#", i + 1, "total", len(files), "files in shard")
        for file in files[:5]:
            print(file)
        print("...")

    if args.gcp:
        worker = parquet_to_arrayrecord_gcp

    else:
        worker = parquet_to_arrayrecord

    os.makedirs(args.output, exist_ok=True)

    with mp.Pool(16) as pool:
        worker = partial(
            worker,
            protocol="hf",
            output_path=args.output,
            prefix="data",
            max_file=len(filenames),
        )

        for _ in pool.imap_unordered(worker, filenames):
            pass
