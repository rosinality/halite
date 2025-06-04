import argparse
import os
import uuid
import time

import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from slickconf import instantiate, load_config

from halite.distributed import load_checkpoint, find_free_port
from halite.projects.common.config import load_model
from halite.transformers.tokainfer.inference_engine import InferenceEngine
from halite.utils import get_torch_dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    conf = load_config(args.conf)
    conf_ckpt = load_model(args.ckpt)

    engine = InferenceEngine(
        instantiate(conf.model_infer)(conf.model), conf_ckpt.tokenizer
    )
    engine.initialize(master_port=find_free_port())

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    backend = "cpu:gloo,cuda:nccl"
    device_id = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(torch.cuda.device(local_rank))
    world_size = int(os.environ["WORLD_SIZE"])

    print("init process group")
    dist.init_process_group(backend, device_id=device_id, world_size=world_size)

    print("instantiate model")
    with torch.device("meta"):
        model = instantiate(conf.model)

    model = model.to(dtype=get_torch_dtype(conf.model_conf.dtype))
    model.to_empty(device=device_id)

    print("load checkpoint")
    load_checkpoint(args.ckpt, model_parts=model)

    print("load state dict")
    engine.load_state_dict(model.state_dict())

    prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think> <answer> answer here </answer>.
User: Find $\begin{pmatrix} 2 \\ -5 \end{pmatrix} - 4 \begin{pmatrix} -1 \\ 7 \end{pmatrix}.$
Assistant: <think>"""

    print("infer batch")
    outputs = engine.infer_batch(
        [
            (prompt, {"temperature": 0, "top_p": 1.0, "n": 2}),
            ("Prove why 1 + 2 = 3", {"temperature": 0, "top_p": 1.0, "n": 2}),
        ]
    )

    tokenizer = instantiate(conf_ckpt.tokenizer)

    print(tokenizer.decode(outputs[0].output_ids[0]))

    engine.cleanup()
    engine.initialize()
    engine.load_state_dict(model.state_dict())

    outputs = engine.infer_batch(
        [
            ("Prove why 1 + 1 = 2", {"temperature": 0, "top_p": 1.0, "n": 2}),
            ("Prove why 1 + 2 = 3", {"temperature": 0, "top_p": 1.0, "n": 2}),
        ]
    )

    print(tokenizer.decode(outputs[0].output_ids[0]))

    engine.destroy()
