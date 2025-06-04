import torch.multiprocessing as mp

from halite.transformers.tokainfer.types import ProcessInfo, ServerConfig
from halite.transformers.tokainfer.basic_worker import (
    start_basic_model_worker,
    start_fanout_worker,
)


def get_pipeline_model_process_dict(
    config: ServerConfig,
    q_manager_to_model: mp.Queue,
    q_model_to_manager: mp.Queue,
    dp_rank: int,
    master_port: int,
):
    pp_size = config.pp_size
    tp_size = config.tp_size
    gpus_per_replica = pp_size * tp_size

    input_qs = [mp.Queue() for _ in range(gpus_per_replica)]
    qs_pipe_end_to_start = [mp.Queue() for _ in range(tp_size)]

    process_dict = {}

    for pp_rank in range(config.pp_size):
        for tp_rank in range(config.tp_size):
            input_q = input_qs[pp_rank * tp_size + tp_rank]
            q_pipe_end_to_start = qs_pipe_end_to_start[tp_rank]

            worker_pinfo = ProcessInfo(
                target=start_pipeline_worker,
                kwargs={
                    "config": config,
                    "input_q": input_q,
                    "q_pipe_end_to_start": q_pipe_end_to_start,
                    "q_to_manager": q_model_to_manager,
                    "pp_rank": pp_rank,
                    "tp_rank": tp_rank,
                    "dp_rank": dp_rank,
                    "master_port": master_port,
                },
            )

            if pp_size > 1 and tp_size > 1:
                name = f"model_worker_pp{pp_rank}_tp{tp_rank}"
            elif pp_size > 1:
                name = f"model_worker_pp{pp_rank}"
            elif tp_size > 1:
                name = f"model_worker_tp{tp_rank}"
            else:
                raise ValueError("Shouldn't happen")

            process_dict[name] = worker_pinfo

    leader_process = ProcessInfo(
        target=start_fanout_worker,
        kwargs={
            "config": config,
            "input_q": q_manager_to_model,
            "fanout_qs": input_qs,
        },
    )

    process_dict["fanout_worker"] = leader_process

    return process_dict


def get_basic_model_process_dict(
    model_conf,
    config: ServerConfig,
    q_manager_to_model: mp.Queue,
    q_model_to_manager: mp.Queue,
    local_rank: int,
    rank: int,
    group_ranks: list[int],
    master_addr: str,
    master_port: int,
    world_size: int,
):
    q_server_to_model = mp.Queue()

    process_info = ProcessInfo(
        target=start_basic_model_worker,
        kwargs={
            "model_conf": model_conf,
            "config": config,
            "input_q": q_manager_to_model,
            "q_model_to_manager": q_model_to_manager,
            "local_rank": local_rank,
            "rank": rank,
            "group_ranks": group_ranks,
            "master_addr": master_addr,
            "master_port": master_port,
            "world_size": world_size,
            "q_server_to_model": q_server_to_model,
        },
    )

    return {
        "model_worker": process_info,
    }, [q_server_to_model]


def get_tp_model_process_dict(
    config: ServerConfig,
    q_manager_to_model: mp.Queue,
    q_model_to_manager: mp.Queue,
    dp_rank: int,
    master_port: int,
):
    process_dict = {}

    input_qs = [mp.Queue() for _ in range(config.tp_size)]

    for tp_rank in range(config.tp_size):
        process_dict[f"model_worker_tp{tp_rank}"] = ProcessInfo(
            target=start_basic_model_worker,
            kwargs={
                "config": config,
                "input_q": input_qs[tp_rank],
                "q_model_to_manager": q_model_to_manager,
                "dp_rank": dp_rank,
                "tp_rank": tp_rank,
                "master_port": master_port,
            },
        )

    process_dict["fanout_worker"] = ProcessInfo(
        target=start_fanout_worker,
        kwargs={
            "config": config,
            "input_q": q_manager_to_model,
            "fanout_qs": input_qs,
        },
    )

    return process_dict


def get_model_process_dict(
    config: ServerConfig,
    q_manager_to_model: mp.Queue,
    q_model_to_manager: mp.Queue,
    dp_rank: int,
    master_port: int,
):
    if config.pp_size > 1:
        return get_pipeline_model_process_dict(
            config=config,
            q_manager_to_model=q_manager_to_model,
            q_model_to_manager=q_model_to_manager,
            dp_rank=dp_rank,
            master_port=master_port,
        )
    elif config.tp_size > 1:
        return get_tp_model_process_dict(
            config=config,
            q_manager_to_model=q_manager_to_model,
            q_model_to_manager=q_model_to_manager,
            dp_rank=dp_rank,
            master_port=master_port,
        )
    else:
        return get_basic_model_process_dict(
            config=config,
            q_manager_to_model=q_manager_to_model,
            q_model_to_manager=q_model_to_manager,
            dp_rank=dp_rank,
            master_port=master_port,
        )
