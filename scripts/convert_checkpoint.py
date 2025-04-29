import argparse
from glob import glob
import os
import json
import shutil

import torch
import torch.distributed.checkpoint as dcp

try:
    from safetensors import safe_open

except ImportError:
    safe_open = None

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

    checkpoint_files = sorted(glob(args.checkpoint_pattern))
    logger.info(f"loading {len(checkpoint_files)} checkpoints")

    checkpoints = []
    for file in checkpoint_files:
        if file.endswith(".safetensors"):
            checkpoint = {}

            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    checkpoint[key] = f.get_tensor(key)

            checkpoints.append(checkpoint)

        else:
            checkpoints.append(torch.load(file, map_location="cpu", weights_only=True))

    logger.info("converting checkpoints")
    converted = convert_checkpoint(
        checkpoints, conf.model_conf, conf.policy, mode="to_halite"
    )

    os.makedirs(args.out, exist_ok=True)

    logger.info("saving config")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(conf.to_dict(), f, indent=2)

    logger.info("saving checkpoints")
    storage_writer = dcp.filesystem.FileSystemWriter(args.out, thread_count=8)
    dcp.save({"model": converted}, storage_writer=storage_writer)

    logger.info("saving tokenizer")
    if hasattr(conf, "tokenizer") and conf.tokenizer is not None:
        with open(os.path.join(args.out, "tokenizer_config.json"), "w") as f:
            json.dump(conf.tokenizer.to_dict(), f, indent=2)

    if args.tokenizer is not None:
        shutil.copy(args.tokenizer, os.path.join(args.out, "tokenizer.model"))
