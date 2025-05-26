import argparse
import os
import json

from array_record.python.array_record_module import ArrayRecordWriter
from datasets import load_dataset, DatasetDict
import orjson
from tqdm import tqdm

from halite.projects.common.template import get_render_fn


TEMPLATE_NO_SPLITS = """from slickconf import field

conf = field(
    root="{{ dataset.root }}",
    ratios={{ dataset.ratios | pformat }},
    shards={{ dataset.shards | pformat }},
)
"""

TEMPLATE_SPLITS = """from slickconf import field

{% for split_name, split in dataset.items()  %}
{{ split_name }} = field(
    root="{{ split.root }}",
    ratios={{ split.ratios | pformat }},
    shards={{ split.shards | pformat }},
)

{% endfor %}

conf = field(
{% for split_name, split in dataset.items()  %}
    {{ split_name }}={{ split_name }},
{% endfor %}
)
"""


def pformat_dict(input):
    formatted = json.dumps(input, indent=4)
    lines = formatted.splitlines()
    indented = []
    for i, line in enumerate(lines):
        if i == 0:
            indented.append(line)

        else:
            indented.append(" " * 4 + line)

    return "\n".join(indented)


def write_arrayrecord(
    dset, dataset_name, name, split, batch_size, n_shards, output_path
):
    dataset_name = dataset_name.replace("/", "_")

    os.makedirs(os.path.join(output_path, dataset_name), exist_ok=True)

    if split is None and isinstance(dset, DatasetDict):
        splits = list(dset.keys())
        dset_dict = True

    else:
        splits = [split]
        dset_dict = False

    dataset_conf = {}

    for split in splits:
        split_conf = {
            "root": output_path,
            "ratios": {dataset_name: 1.0},
            "shards": {dataset_name: {}},
        }

        if name is None:
            filename = f"{split}-{{i}}-of-{n_shards}.arrayrecord"

        else:
            filename = f"{name}-{split}-{{i}}-of-{n_shards}.arrayrecord"

        shard_names = [filename.format(i=i + 1) for i in range(n_shards)]
        shard_writer = [
            ArrayRecordWriter(
                os.path.join(output_path, dataset_name, name), "group_size:1"
            )
            for name in shard_names
        ]

        if dset_dict:
            dset_iter = dset[split].iter(batch_size=batch_size)

        else:
            dset_iter = dset.iter(batch_size=batch_size)

        total_samples = [0] * n_shards

        for i, batch in enumerate(dset_iter):
            records = []
            keys = list(batch.keys())

            for row in zip(*[batch[key] for key in keys]):
                records.append({key: value for key, value in zip(keys, row)})

            for record in records:
                shard_writer[i % n_shards].write(orjson.dumps(record))
                total_samples[i % n_shards] += 1

        for writer in shard_writer:
            writer.close()

        for split_name, total in zip(shard_names, total_samples):
            split_conf["shards"][dataset_name][split_name] = total

        dataset_conf[split] = split_conf

    return dataset_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--names", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("dataset", type=str)

    args = parser.parse_args()

    if args.names is not None:
        names = args.names.split(",")

    else:
        names = [None]

    dsets = []
    for name in names:
        dsets.append(
            load_dataset(
                args.dataset,
                name,
                split=args.split,
                trust_remote_code=args.trust_remote_code,
            )
        )

    os.makedirs(args.output, exist_ok=True)

    dataset_conf = None

    for name, dset in tqdm(zip(names, dsets)):
        print(args.dataset, name, len(dset))
        dset_conf = write_arrayrecord(
            dset,
            args.dataset,
            name,
            args.split,
            args.batch_size,
            args.n_shards,
            args.output,
        )

        if dataset_conf is None:
            dataset_conf = dset_conf

        else:
            for split, conf in dataset_conf.items():
                for dataset_name, shard_conf in conf["shards"].items():
                    shard_conf.update(dset_conf[split]["shards"][dataset_name])

    if len(dataset_conf) == 1:
        dataset_conf = dataset_conf[next(dataset_conf.keys())]
        render_fn = get_render_fn(TEMPLATE_NO_SPLITS, filters={"pformat": pformat_dict})

    else:
        render_fn = get_render_fn(TEMPLATE_SPLITS, filters={"pformat": pformat_dict})

    dataset_name = dataset_name.replace("/", "_")

    with open(os.path.join(args.output, dataset_name, "dataset.py"), "w") as f:
        f.write(render_fn(dataset=dataset_conf))

    with open(os.path.join(args.output, dataset_name, "dataset.json"), "wb") as f:
        f.write(orjson.dumps(dataset_conf, option=orjson.OPT_INDENT_2))
