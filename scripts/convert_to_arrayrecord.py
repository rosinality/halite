import argparse
from glob import glob
import os
import math
import mmap
import multiprocessing as mp

from array_record.python.array_record_module import ArrayRecordWriter


def read_worker(file_name, writer_queues, n_shards):
    file_size = os.path.getsize(file_name)
    bytes_total_gb = round(file_size / (1024**3), 2)
    bytes_read = 0

    with open(file_name, "r") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        for i, line in enumerate(iter(mm.readline, b"")):
            writer_queues[i % n_shards % len(writer_queues)].put((i % n_shards, line))
            bytes_read += len(line)

            if i > 0 and i % 10000 == 0:
                bytes_gb = round(bytes_read / (1024**3), 2)
                print(
                    f"{os.path.basename(file_name)} - {i}, {bytes_gb}/{bytes_total_gb} GB"
                )

        bytes_gb = round(bytes_read / (1024**3), 2)
        print(
            f"[finished] {os.path.basename(file_name)} - {i}, {bytes_gb}/{bytes_total_gb} GB"
        )

        for queue in writer_queues:
            queue.put((None, None))

        mm.close()


def write_worker(output_path, reader_queue, worker_id, n_worker, n_shards):
    shard_target = set(
        [
            i % n_shards
            for i in range(n_shards * n_worker)
            if i % n_shards % n_worker == worker_id
        ]
    )

    shard_writer = {
        k: ArrayRecordWriter(
            os.path.join(output_path, f"data-{k + 1}-of-{n_shards}.arrayrecord"),
            "group_size:1",
        )
        for k in shard_target
    }

    while True:
        shard, line = reader_queue.get()
        if shard is None:
            break

        shard_writer[shard].write(line)

    for writer in shard_writer.values():
        writer.close()


def process_files(plan, path_prefix, output_path):
    reader_procs = []
    writer_procs = []

    for fname, n_shards, _, shard_per_writer in plan:
        writer_queues = [mp.Queue() for _ in range(shard_per_writer)]
        reader_procs.append(
            mp.Process(target=read_worker, args=(fname, writer_queues, n_shards))
        )

        relpath = os.path.splitext(os.path.relpath(fname, path_prefix))[0]
        target_path = os.path.join(output_path, relpath)
        os.makedirs(target_path, exist_ok=True)

        for writer_id, writer_queue in enumerate(writer_queues):
            writer_procs.append(
                mp.Process(
                    target=write_worker,
                    args=(
                        target_path,
                        writer_queue,
                        writer_id,
                        len(writer_queues),
                        n_shards,
                    ),
                )
            )

    for p in reader_procs:
        p.start()

    for p in writer_procs:
        p.start()

    for p in reader_procs:
        p.join()

    for p in writer_procs:
        p.join()

    for p in reader_procs:
        p.close()

    for p in writer_procs:
        p.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    parser.add_argument("--max_writer", type=int, default=32)
    parser.add_argument("--shard_size", type=int, default=1)
    parser.add_argument("input", type=str)

    args = parser.parse_args()

    files = glob(os.path.join(args.input, "**/*.jsonl"), recursive=True)
    files = {key: os.path.getsize(key) for key in files}
    shard_byte = args.shard_size * (1024**3)

    n_shards = {key: math.ceil(value / shard_byte) for key, value in files.items()}
    n_shards_sorted = sorted(n_shards.items(), key=lambda x: x[1], reverse=False)
    plan = []

    max_writer = args.max_writer
    assigned_writer = 0
    plan_step = []
    for key, value in n_shards_sorted:
        avail_writer = max_writer - assigned_writer
        plan_step.append((key, value, avail_writer, math.ceil(value / avail_writer)))
        assigned_writer += value

        if assigned_writer >= max_writer:
            plan.append(plan_step)
            plan_step = []
            assigned_writer = 0

    for step in plan:
        process_files(step, args.input, args.output)
