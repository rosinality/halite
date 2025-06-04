import argparse
import os
import uuid
import time

import torch
from torch import distributed as dist
from torch.distributed.distributed_c10d import is_initialized, _get_default_group
import torch.multiprocessing as mp
from slickconf import instantiate, load_config

from halite.distributed import load_checkpoint, init_custom_process_group
from halite.projects.common.config import load_model
from halite.transformers.tokainfer.entry import get_basic_model_process_dict
from halite.transformers.tokainfer.types import (
    ServerConfig,
    ProcessInfo,
    UpdateStateDict,
)
from halite.transformers.tokainfer.types import SamplingParams, TokasaurusRequest
from halite.transformers.tokainfer.engine.manager import start_manager
from halite.utils import get_torch_dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    q_server_to_manager = mp.Queue()
    q_manager_to_server = mp.Queue()

    q_manager_to_model = mp.Queue()
    q_model_to_manager = mp.Queue()

    q_server_to_model = mp.Queue()

    conf = load_config(args.conf)
    conf_ckpt = load_model(args.ckpt)
    tokenizer = instantiate(conf_ckpt.tokenizer)
    server_conf = ServerConfig()

    barrier = mp.Barrier(3)

    world_size = int(os.environ["WORLD_SIZE"])

    manager_proc = ProcessInfo(
        target=start_manager,
        kwargs={
            "tokenizer": conf_ckpt.tokenizer,
            "config": server_conf,
            "q_manager_to_model": q_manager_to_model,
            "q_model_to_manager": q_model_to_manager,
            "q_server_to_manager": q_server_to_manager,
            "q_manager_to_server": q_manager_to_server,
            "process_name": "manager",
            "barrier": barrier,
        },
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    backend = "gloo"
    device_id = None

    backend = "cpu:gloo,cuda:nccl"
    device_id = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(torch.cuda.device(local_rank))

    model_dict = get_basic_model_process_dict(
        instantiate(conf.model_infer)(conf.model),
        server_conf,
        q_manager_to_model,
        q_model_to_manager,
        master_addr=os.environ["MASTER_ADDR"],
        master_port=os.environ["MASTER_PORT"],
        world_size=world_size * 2,
        local_rank=local_rank + 1,
        rank=rank + world_size,
        group_ranks=[rank, rank + world_size],
        q_server_to_model=q_server_to_model,
    )

    model_dict["model_worker"].kwargs["barrier"] = barrier
    model_dict["model_worker"].kwargs["process_name"] = "model_worker"

    p = manager_proc.make_process()
    p.start()

    p = model_dict["model_worker"].make_process()
    p.start()

    dist.init_process_group(backend, device_id=device_id, world_size=world_size * 2)
    print([rank, rank + world_size])
    pg = dist.new_group([rank, rank + world_size])
    train_pg = dist.new_group([rank])

    print("initialized", is_initialized())
    print("default group", _get_default_group())

    with torch.device("meta"):
        model = instantiate(conf.model)

    print("model to empty")

    model = model.to(dtype=get_torch_dtype(conf.model_conf.dtype))
    model.to_empty(device=device_id)

    print("load checkpoint")

    load_checkpoint(args.ckpt, model_parts=model, process_group=train_pg)

    barrier.wait()

    print("update call")

    q_server_to_manager.put(UpdateStateDict(method="distributed"))

    state_dict = model.state_dict()
    keys = sorted(state_dict.keys())

    works = []

    print("server start update dict")

    for key in keys:
        tensor = state_dict[key]
        # q_server_to_model.put(tensor)
        # dist.send(tensor, group=train_infer_group, group_dst=rank + world_size)
        work = dist.broadcast(tensor, group=pg, group_src=0, async_op=True)
        works.append(work)
        # dist.broadcast(tensor, group=train_infer_group, group_src=0)
        # train_infer_group.broadcast(tensor, 0)

    [work.wait() for work in works]

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0)
    input_ids = tokenizer.encode("Prove why 1 + 1 = 2")
    rid = str(uuid.uuid4())
    req = TokasaurusRequest(
        id=rid,
        input_ids=input_ids,
        max_num_tokens=128,
        sampling_params=sampling_params,
        stop=[],
        n=1,
        ignore_eos=False,
    )
    q_server_to_manager.put(req)

    while True:
        if not q_manager_to_server.empty():
            out = q_manager_to_server.get()

            print(out)
            print(tokenizer.decode(out.completion_ids[0]))

            break

        time.sleep(0.01)

    p.join()
    p.kill()
