import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from slickconf import instantiate, load_arg_config, summarize

from halite.data.dataloader import DataManager
from halite.data.dataset import build_dataset_from_spec, WeightedIterableDataset
from halite.logging import get_logger
from halite.parallel import ParallelDims


def detokenize(input, tokenizer):
    return tokenizer.decode([token for token in input if token >= 0])


def log_samples(
    n_batch,
    train_loader,
    tokenizer,
    parallel_dims,
    logger,
    output_path,
):
    loader = iter(DataManager(train_loader, parallel_dims.mesh))

    step = 0
    batches = []
    while True:
        try:
            batch = next(loader)

        except StopIteration:
            break

        logger.info(f"batch {step}")
        logger.info(batch)

        batch_keys = batch._meta_["tokenized_keys"]
        for key in batch_keys:
            feature = batch[key].tolist()

            logger.info(key)
            for tokens in feature:
                logger.info(detokenize(tokens, tokenizer))

        batches.append(batch)

        step += 1

        if step >= n_batch:
            break

    if output_path is not None:
        with open(output_path, "wb") as f:
            pickle.dump(batches, f)

    return step


def get_parallel_size(conf, name, default=1):
    if "training" not in conf:
        return default

    return conf.training.get(name, default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_batch", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    conf, args = load_arg_config(config_model=None, parser=parser)
    tokenizer = instantiate(conf.model.tokenizer)
    del conf.model

    pdims = ParallelDims(
        dp_replicate=get_parallel_size(conf, "data_parallel_replicate"),
        dp_shard=get_parallel_size(conf, "data_parallel_shard", -1),
        tp=get_parallel_size(conf, "tensor_parallel"),
        pp=get_parallel_size(conf, "pipeline_parallel"),
    )
    mesh = pdims.build_mesh("cuda")
    logger = get_logger(mesh)

    logger.info(summarize(conf))
    logger.info(
        f"dp replicate: {pdims.dp_replicate}, dp shard: {pdims.dp_shard}, tp: {pdims.tp} pp: {pdims.pp}"
    )

    torch.distributed.barrier()

    train_source, train_ratios, train_names = build_dataset_from_spec(
        conf.data.train, split="train", split_ratio=conf.data.get("train_ratio", 1.0)
    )

    preprocess_ops = []
    if conf.data.preprocess is not None:
        for op in conf.data.preprocess:
            preprocess_ops.append(instantiate(op))

    collate_fn = None
    if conf.data.get("collate_fn", None) is not None:
        collate_fn = instantiate(conf.data.collate_fn)

    train_dset = WeightedIterableDataset(
        train_source,
        train_ratios,
        train_names,
        num_replicas=pdims.dp,
        rank=mesh.get_local_rank("dp"),
        operations=preprocess_ops,
    )

    logger.info(f"train_dset\n{train_dset.summary()}")

    train_loader = DataLoader(
        train_dset,
        batch_size=conf.training.train_batch_size // pdims.dp,
        collate_fn=collate_fn,
        num_workers=2,
    )

    log_samples(args.n_batch, train_loader, tokenizer, pdims, logger, args.output)


if __name__ == "__main__":
    main()

    torch.distributed.destroy_process_group()
