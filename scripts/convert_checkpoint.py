import argparse
from glob import glob
import os
import json
import shutil

import torch
import torch.distributed.checkpoint as dcp
from slickconf import load_config

from halite.logging import logger
from halite.transformers.convert import convert_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--out", type=str)
    parser.add_argument("checkpoint_pattern", type=str)

    args = parser.parse_args()

    conf = load_config(args.conf)

    checkpoints = sorted(glob(args.checkpoint_pattern))
    logger.info(f"loading {len(checkpoints)} checkpoints")
    checkpoints = [
        torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        for checkpoint in checkpoints
    ]

    logger.info("converting checkpoints")
    converted = convert_checkpoint(
        checkpoints, conf.model_conf, conf.policy, mode="to_halite"
    )

    logger.info("saving config")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(conf.to_dict(), f, indent=2)

    logger.info("saving checkpoints")
    os.makedirs(args.out, exist_ok=True)
    storage_writer = dcp.filesystem.FileSystemWriter(args.out, thread_count=8)
    dcp.save({"model": converted}, storage_writer=storage_writer)

    logger.info("saving tokenizer")
    if conf.tokenizer is not None:
        with open(os.path.join(args.out, "tokenizer_config.json"), "w") as f:
            json.dump(conf.tokenizer.to_dict(), f, indent=2)

    if args.tokenizer is not None:
        shutil.copy(args.tokenizer, os.path.join(args.out, "tokenizer.model"))
