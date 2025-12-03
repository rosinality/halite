import argparse
from contextlib import contextmanager
from glob import glob
import os
import io
import math
import mmap
import multiprocessing as mp

try:
    from array_record.python.array_record_module import ArrayRecordWriter

except ImportError:
    ArrayRecordWriter = None


try:
    from ffrecord import FileWriter

except ImportError:
    FileWriter = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

except ImportError:
    pa = None
    pq = None

try:
    import zstandard

except ImportError:
    zstandard = None


def convert_to_bytes(iterator):
    try:
        for line in iterator:
            yield line.encode("utf-8")

    except zstandard.ZstdError as e:
        yield None


@contextmanager
def get_read_iterator(stream, format):
    if format == "text":
        yield iter(stream.readline, b"")

    elif format == "zstd":
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(stream) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            yield convert_to_bytes(text_stream)


def read_file(file_name, writer_queues, n_shards):
    file_size = os.path.getsize(file_name)
    bytes_total_gb = round(file_size / (1024**3), 2)
    bytes_read = 0

    with open(file_name, "r") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        format = "zstd" if file_name.endswith(".zstd") else "text"

        with get_read_iterator(mm, format) as reader:
            for i, line in enumerate(reader):
                if line is None:
                    continue

                writer_queues[i % n_shards % len(writer_queues)].put(
                    (i % n_shards, line)
                )
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

        mm.close()


def read_worker(filenames, writer_queues, n_shards):
    for filename in filenames:
        read_file(filename, writer_queues, n_shards)

    for queue in writer_queues:
        queue.put((None, None))


def write_worker(
    output_path,
    reader_queue,
    worker_id,
    n_worker,
    n_shards,
    format,
    start_offset,
    total_shards,
):
    shard_target = set(
        [
            i % n_shards
            for i in range(n_shards * n_worker)
            if i % n_shards % n_worker == worker_id
        ]
    )

    if format == "arrayrecord":
        shard_writer = {
            k: ArrayRecordWriter(
                os.path.join(
                    output_path,
                    f"data-{k + 1 + start_offset}-of-{total_shards}.arrayrecord",
                ),
                "group_size:1",
            )
            for k in shard_target
        }

        while True:
            shard, line = reader_queue.get()
            if shard is None:
                break

            shard_writer[shard].write(line)

    elif format == "parquet":
        schema = pa.schema([("data", pa.binary())])

        shard_writer = {
            k: pq.ParquetWriter(
                os.path.join(
                    output_path,
                    f"data-{k + 1 + start_offset}-of-{total_shards}.parquet",
                ),
                schema=schema,
            )
            for k in shard_target
        }
        shard_buffer = {k: [] for k in shard_target}

        while True:
            shard, line = reader_queue.get()
            if shard is None:
                break

            shard_buffer[shard].append(line)

            if len(shard_buffer[shard]) >= 1024:
                shard_writer[shard].write(
                    pa.record_batch([shard_buffer[shard]], names=schema.names)
                )
                shard_buffer[shard] = []

        for shard in shard_target:
            if len(shard_buffer[shard]) > 0:
                shard_writer[shard].write(
                    pa.record_batch([shard_buffer[shard]], names=schema.names)
                )

    elif format == "ffrecord":
        shard_buffer = {k: [] for k in shard_target}

        while True:
            shard, line = reader_queue.get()
            if shard is None:
                break

            shard_buffer[shard].append(line)

        shard_writer = {
            k: FileWriter(
                os.path.join(
                    output_path,
                    f"data-{k + 1 + start_offset}-of-{total_shards}.ffr",
                ),
                len(v),
            )
            for k, v in shard_buffer.items()
        }

        for key, buffer in shard_buffer.items():
            for line in buffer:
                shard_writer[key].write_one(line)

    for writer in shard_writer.values():
        writer.close()


def process_files(plan, path_prefix, output_path, format, start_offset, total_shards):
    reader_procs = []
    writer_procs = []

    start_offset_label = start_offset

    for fnames, n_shards, _, shard_per_writer in plan:
        writer_queues = [mp.Queue() for _ in range(shard_per_writer)]
        reader_procs.append(
            mp.Process(target=read_worker, args=(fnames, writer_queues, n_shards))
        )

        if len(fnames) > 1 or n_shards == 1:
            target_path = output_path
            os.makedirs(target_path, exist_ok=True)
            start_offset_label = start_offset
            total_shards_label = total_shards

        else:
            relpath = os.path.splitext(os.path.relpath(fname, path_prefix))[0]
            target_path = os.path.join(output_path, relpath)
            os.makedirs(target_path, exist_ok=True)
            start_offset_label = 0
            total_shards_label = n_shards

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
                        format,
                        start_offset_label,
                        total_shards_label,
                    ),
                )
            )

        if len(fnames) > 1 or n_shards == 1:
            start_offset += n_shards

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
    parser.add_argument("--target_ext", type=str, default="jsonl")
    parser.add_argument("--format", type=str, default="arrayrecord")
    parser.add_argument("input", type=str)

    args = parser.parse_args()

    files = glob(os.path.join(args.input, f"**/*.{args.target_ext}"), recursive=True)
    files = {key: os.path.getsize(key) for key in files}
    shard_byte = args.shard_size * (1024**3)

    shards = []
    sub_shards = []
    current_size = 0
    for fname, size in files.items():
        if current_size + size > shard_byte:
            if len(sub_shards) > 0:
                shards.append((sub_shards, 1))
                sub_shards = []

            current_size = 0

        if size >= shard_byte:
            shards.append(([fname], math.ceil(size / shard_byte)))

            continue

        sub_shards.append(fname)
        current_size += size

    if len(sub_shards) > 0:
        shards.append((sub_shards, 1))

    plan = []
    max_writer = args.max_writer
    assigned_writer = 0
    plan_step = []
    for subshards, n_shard in shards:
        avail_writer = max_writer - assigned_writer
        plan_step.append(
            (subshards, n_shard, avail_writer, math.ceil(n_shard / avail_writer))
        )
        assigned_writer += n_shard

        if assigned_writer >= max_writer:
            plan.append(plan_step)
            plan_step = []
            assigned_writer = 0

    if len(plan_step) > 0:
        plan.append(plan_step)

    total_shards = sum([shard[1] for step in plan for shard in step])

    start_offset = 0
    for step in plan:
        process_files(
            step, args.input, args.output, args.format, start_offset, total_shards
        )
        start_offset += len(step)
