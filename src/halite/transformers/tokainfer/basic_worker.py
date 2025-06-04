from dataclasses import dataclass

import torch
from torch import nn
import torch.multiprocessing as mp
from torch import Tensor

from halite.transformers.tokainfer.types import (
    ServerConfig,
    TimedBarrier,
)
from halite.transformers.tokainfer.types import (
    BasicWorkerState,
    BatchState,
    CommandFromManager,
    Initialize,
    Cleanup,
    ModelInput,
    ModelOutput,
    NoMoreInputs,
    UpdateStateDict,
)
from halite.transformers.tokainfer.utils import (
    ModelRunner,
    add_decoding_ids_to_batch_state,
    get_dtype,
    make_input_batch_state,
    make_model,
    move_batch_state,
    setup_and_run_loop,
    setup_distributed,
    unpad_output_batch_state,
)


def basic_model_loop(
    state: BasicWorkerState,
    model: nn.Module,
    q_server_to_model: mp.Queue,
    device: str | torch.device,
    worker_barrier: TimedBarrier,
):
    tp_rank = 0
    tp_size = 1
    non_blocking = True

    @dataclass
    class Work:
        model_input: ModelInput
        input_batch_state: BatchState
        batch_indices: Tensor
        output_batch_state: BatchState | None = None
        output_tokens_cpu: Tensor | None = None
        logprobs_cpu: Tensor | None = None

    def preprocess():
        while True:
            command: CommandFromManager = state.input_q.get()

            match command:
                case ModelInput():
                    inp = command

                    break

                case UpdateStateDict():
                    model_runner.update_state_dict(command)

                    continue

                case Initialize():
                    state.initialize()
                    model_runner.initialize()

                    continue

                case Cleanup():
                    state.cleanup()
                    model_runner.cleanup()

                    continue

                case NoMoreInputs():
                    return None

                case _:
                    raise ValueError(f"Unknown command: {type(command)}")

        batch_indices = torch.tensor(
            inp.batch_indices,
            dtype=torch.long,
        )

        input_batch_state = make_input_batch_state(
            inp,
            tp_rank=tp_rank,
            tp_size=tp_size,
            add_raw_lm_head_indices=tp_size > 1,
        )

        model_runner.plan(input_batch_state, non_blocking=non_blocking)

        move_batch_state(
            input_batch_state=input_batch_state,
            device=state.device,
            non_blocking=non_blocking,
        )

        return Work(
            model_input=inp,
            input_batch_state=input_batch_state,
            batch_indices=batch_indices.to(state.device, non_blocking=non_blocking),
        )

    def run_model(work: Work):
        decoding_batch_indices = work.batch_indices[
            work.model_input.decode_start_pos() :
        ]
        decoding_input_ids = state.batch_index_to_last_token[decoding_batch_indices]

        input_batch_state = work.input_batch_state

        add_decoding_ids_to_batch_state(
            input_batch_state=input_batch_state,
            decoding_input_ids=decoding_input_ids,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        output_batch_state = model_runner.run(
            input_batch_state, non_blocking=non_blocking
        )

        unpad_output_batch_state(
            output_batch_state=output_batch_state,
            input_batch_state=input_batch_state,
        )

        if tp_size > 1:
            lm_head_indices = input_batch_state.raw_lm_head_indices
        else:
            lm_head_indices = input_batch_state.lm_head_indices

        assert lm_head_indices is not None
        batch_indices = work.batch_indices[lm_head_indices]

        if len(batch_indices) > 0:
            assert output_batch_state.output_ids is not None
            state.batch_index_to_last_token[batch_indices] = (
                output_batch_state.output_ids
            )

        work.output_batch_state = output_batch_state

    def synchronize(work: Work):
        # technically, we don't need to sync when tp_rank != 0,
        # but omitting it causes sporadic nccl illegal memory access errors
        torch.cuda.synchronize()

        work.output_tokens_cpu = work.output_batch_state.output_ids.cpu()
        work.logprobs_cpu = work.output_batch_state.logprobs.cpu()

    def postprocess(work: Work):
        assert work.output_tokens_cpu is not None
        assert work.logprobs_cpu is not None

        model_input = work.model_input
        out = ModelOutput(
            output_tokens=work.output_tokens_cpu.tolist(),
            logprobs=work.logprobs_cpu.tolist(),
            schedule_id=model_input.schedule_id,
        )

        state.q_model_to_manager.put(out)

    model_runner = ModelRunner(
        config=state.config,
        model=model,
        process_group=state.process_group,
        q_server_to_model=q_server_to_model,
        device=device,
    )

    setup_and_run_loop(
        state=state,
        model_runner=model_runner,
        preprocess=preprocess,
        run_model=run_model,
        synchronize=synchronize,
        postprocess=postprocess,
    )


def start_basic_model_worker(
    model_conf,
    config: ServerConfig,
    input_q: mp.Queue,
    q_model_to_manager: mp.Queue,
    rank: int,
    local_rank: int,
    group_ranks: list[int],
    process_name: str,
    barrier: TimedBarrier,
    master_addr: str,
    master_port: int,
    world_size: int,
    q_server_to_model: mp.Queue,
    worker_barrier: TimedBarrier,
):
    process_group, device_mesh, device = setup_distributed(
        config,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
        world_size=world_size,
        group_ranks=group_ranks,
    )
    dtype = get_dtype(config.dtype)

    batch_index_to_last_token = torch.zeros(
        config.max_batch_index(), dtype=torch.long, device=device
    )

    state = BasicWorkerState(
        config=config,
        batch_index_to_last_token=batch_index_to_last_token,
        input_q=input_q,
        q_model_to_manager=q_model_to_manager,
        device=device,
        dtype=dtype,
        process_name=process_name,
        rank=rank,
        local_rank=local_rank,
        barrier=barrier,
        process_group=process_group,
    )

    model = make_model(
        model_conf,
        config,
        device,
        dtype,
    )

    basic_model_loop(state, model, q_server_to_model, device, worker_barrier)


def start_fanout_worker(
    input_q: mp.Queue,
    fanout_qs: list[mp.Queue],
    barrier: TimedBarrier,
):
    barrier.wait()

    while True:
        inp = input_q.get()
        for q in fanout_qs:
            q.put(inp)
