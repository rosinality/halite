import cProfile
import multiprocessing.connection as mp_conn
import pickle
import queue
import socket
import subprocess
import gc
import time
import traceback
from contextlib import contextmanager
from functools import wraps
import math
import os
from copy import deepcopy
from dataclasses import dataclass

from slickconf import instantiate
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from flashinfer import cascade
from torch import Tensor
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from tqdm import tqdm
import psutil
import requests
import torch.distributed

from halite.distributed import init_custom_process_group
from halite.transformers.tokainfer.types import ServerConfig
from halite.transformers.tokainfer.engine.input_building import make_dummy_batch
from halite.transformers.tokainfer.attention_fn import create_wrappers_for_cudagraph
from halite.transformers.tokainfer.types import (
    BasicWorkerState,
    BatchState,
    DeviceType,
    ModelInput,
    UpdateStateDict,
    Initialize,
    Cleanup,
    NoMoreInputs,
    PipelineWorkerState,
    WrapperCollection,
)


def get_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "fp8": torch.float8_e4m3fn,
    }
    return dtype_map[dtype_str]


def get_global_rank(config: ServerConfig, dp_rank: int, pp_rank: int, tp_rank: int):
    """
    parallelism order from outer to inner: dp -> pp -> tp
    """
    return (
        dp_rank * config.pp_size * config.tp_size + (pp_rank * config.tp_size) + tp_rank
    )


def setup_distributed(
    config: ServerConfig,
    rank: int,
    local_rank: int,
    master_addr: str,
    master_port: int,
    world_size: int,
    group_ranks: list[list[int]],
):
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    backend = "cpu:gloo,cuda:nccl"
    device_id = torch.device(f"cuda:{local_rank}")

    store = dist.TCPStore(master_addr, master_port, world_size, rank == 0)

    os.environ["RANK"] = str(rank)

    dist.init_process_group(
        backend="nccl",
        device_id=device_id,
        store=store,
        world_size=world_size,
        rank=rank,
    )

    local_pg = None

    for group in group_ranks:
        pg = dist.new_group(group)

        if rank in group:
            local_pg = pg

    device_mesh = None

    return local_pg, device_mesh, device


def last_page_len(length: int, page_size: int):
    last_page_len = length % page_size
    if last_page_len == 0:
        last_page_len = page_size

    return last_page_len


def tp_slice(x: Tensor, tp_rank: int, tp_size: int) -> Tensor:
    bs = x.shape[0]
    assert bs % tp_size == 0, f"bs={bs} must be divisible by tp_size={tp_size}"
    bs_per_rank = bs // tp_size
    tp_start = tp_rank * bs_per_rank
    tp_end = tp_start + bs_per_rank
    return x[tp_start:tp_end]


def pad_and_slice_tensor(
    x: Tensor, num_padding: int, tp_rank: int, tp_size: int
) -> Tensor:
    if num_padding > 0:
        x = F.pad(x, (0, num_padding))

    return tp_slice(x, tp_rank, tp_size)


def make_input_batch_state(
    inp: ModelInput,
    tp_rank: int = 0,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    add_raw_lm_head_indices: bool = False,
):
    skip_input_ids = pp_size > 1 and pp_rank > 0
    skip_lm_head_indices = pp_size > 1 and pp_rank < pp_size - 1

    if skip_input_ids:
        prefill_input_ids = None
    else:
        prefill_input_ids = torch.tensor(
            inp.prefill_input_ids,
            dtype=torch.long,
        )

    position_ids = torch.tensor(inp.position_ids, dtype=torch.long)

    if skip_lm_head_indices:
        lm_head_indices = None
    else:
        lm_head_indices = torch.tensor(inp.lm_head_indices, dtype=torch.long)

    # Build attention_info from the builder
    attention_info = inp.build_attention_info()

    # Build sampling_params from the builder
    sampling_params = inp.build_sampling_params()

    input_batch_state = BatchState(
        prefill_input_ids=prefill_input_ids,
        attention_info=attention_info,
        position_ids=position_ids,
        sampling_params=sampling_params,
        lm_head_indices=lm_head_indices,
        raw_lm_head_indices=lm_head_indices if add_raw_lm_head_indices else None,
    )

    if tp_size > 1:
        assert tp_rank is not None
        bs = input_batch_state.position_ids.shape[0]
        padded_bs = math.ceil(bs / tp_size) * tp_size
        num_padding = padded_bs - bs
        input_batch_state.attention_info.num_padding = num_padding

        if num_padding > 0:
            input_batch_state.position_ids = F.pad(
                input_batch_state.position_ids,
                (0, num_padding),
                value=0,
            )

        if not skip_lm_head_indices:
            assert input_batch_state.lm_head_indices is not None

            num_lm_head_indices = input_batch_state.lm_head_indices.shape[0]
            padded_num_lm_head_indices = (
                math.ceil(num_lm_head_indices / tp_size) * tp_size
            )
            lm_head_indices_padding = padded_num_lm_head_indices - num_lm_head_indices
            input_batch_state.lm_head_indices_padding = lm_head_indices_padding

            input_batch_state.lm_head_indices = pad_and_slice_tensor(
                input_batch_state.lm_head_indices,
                lm_head_indices_padding,
                tp_rank,
                tp_size,
            )

            if (
                greedy_mask := input_batch_state.sampling_params.greedy_mask
            ) is not None:
                input_batch_state.sampling_params.greedy_mask = pad_and_slice_tensor(
                    greedy_mask,
                    lm_head_indices_padding,
                    tp_rank,
                    tp_size,
                )

            if (top_p := input_batch_state.sampling_params.top_p) is not None:
                input_batch_state.sampling_params.top_p = pad_and_slice_tensor(
                    top_p,
                    lm_head_indices_padding,
                    tp_rank,
                    tp_size,
                )

            if (
                temperature := input_batch_state.sampling_params.temperature
            ) is not None:
                input_batch_state.sampling_params.temperature = pad_and_slice_tensor(
                    temperature,
                    lm_head_indices_padding,
                    tp_rank,
                    tp_size,
                )

    return input_batch_state


def add_decoding_ids_to_batch_state(
    input_batch_state: BatchState,
    decoding_input_ids: Tensor,
    tp_rank: int = 0,
    tp_size: int = 1,
):
    assert input_batch_state.prefill_input_ids is not None
    input_batch_state.input_ids = torch.cat(
        [input_batch_state.prefill_input_ids, decoding_input_ids], dim=0
    )

    if tp_size > 1:
        num_padding = input_batch_state.attention_info.num_padding
        if num_padding > 0:
            input_batch_state.input_ids = F.pad(
                input_batch_state.input_ids,
                (0, num_padding),
                value=0,
            )

        input_batch_state.input_ids = tp_slice(
            input_batch_state.input_ids, tp_rank, tp_size
        )

    # don't need it anymore, no need to move it to device
    input_batch_state.prefill_input_ids = None


def move_batch_state(
    input_batch_state: BatchState,
    device: DeviceType,
    non_blocking: bool = False,
):
    input_batch_state.position_ids = input_batch_state.position_ids.to(
        device, non_blocking=non_blocking
    )

    input_batch_state.attention_info.append_kv_token_indices = (
        input_batch_state.attention_info.append_kv_token_indices.to(
            device, non_blocking=non_blocking
        )
    )

    input_batch_state.sampling_params = input_batch_state.sampling_params.to(
        device, non_blocking=non_blocking
    )

    if input_batch_state.lm_head_indices is not None:
        input_batch_state.lm_head_indices = input_batch_state.lm_head_indices.to(
            device, non_blocking=non_blocking
        )

    if input_batch_state.raw_lm_head_indices is not None:
        input_batch_state.raw_lm_head_indices = (
            input_batch_state.raw_lm_head_indices.to(device, non_blocking=non_blocking)
        )

    if input_batch_state.input_ids is not None:
        input_batch_state.input_ids = input_batch_state.input_ids.to(
            device, non_blocking=non_blocking
        )

    if input_batch_state.prefill_input_ids is not None:
        input_batch_state.prefill_input_ids = input_batch_state.prefill_input_ids.to(
            device, non_blocking=non_blocking
        )


def run_overlapped_loop(
    preprocess,
    run_model,
    synchronize,
    postprocess,
    max_iters: int | None = None,
    prog_bar_name: str | None = None,
):
    preproc_work = None
    run_work = None
    postproc_work = None

    iter_num = 0

    if max_iters is not None and prog_bar_name is not None:
        prog_bar = tqdm(total=max_iters, desc=prog_bar_name)
    else:
        prog_bar = None

    while True:
        if preproc_work is not None:
            run_work = preproc_work
            preproc_work = None

        else:
            # non-overlapped preprocess, our goal is to
            # avoid these whenever possible

            run_work = preprocess()

        if run_work is None:
            continue

        run_model(run_work)

        if postproc_work is not None:
            postprocess(postproc_work)
            postproc_work = None

        preproc_work = preprocess()

        synchronize(run_work)
        postproc_work = run_work

        # if we're going to block waiting for the next input, postprocess now
        if preproc_work is None:
            postprocess(postproc_work)
            postproc_work = None

        iter_num += 1

        if prog_bar is not None:
            prog_bar.update(1)

        if max_iters is not None and iter_num >= max_iters:
            break


def make_model(
    model_conf,
    config: ServerConfig,
    device: str,
    dtype: torch.dtype,
    model_parallelize=None,
    mesh=None,
    pdims=None,
    tp_group: dist.ProcessGroup | None = None,
):
    torch.compile(lambda x: x)

    with torch.device("meta"):
        model = instantiate(model_conf)

    if model_parallelize is not None:
        model = instantiate(model_parallelize)(
            model=model, mesh=mesh, paralel_dims=pdims
        )

    model = model.to(dtype=dtype)
    model.to_empty(device=device)

    num_pages = config.kv_cache_num_blocks()

    # extra page to point padding append indices when using cudagraphs
    if config.use_cudagraphs:
        num_pages += 1

    model.setup_caches(num_pages=num_pages, page_size=config.page_size)

    if config.torch_compile:
        use_async_tp = config.tp_size > 1 and config.async_tp_threshold is not None
        if use_async_tp:
            torch._dynamo.config.cache_size_limit = 32
            assert tp_group is not None
            enable_symm_mem_for_group(tp_group.group_name)

        model.forward_blocks = torch.compile(
            model.forward_blocks, fullgraph=True, dynamic=True
        )

    return model


def set_async_tp_enabled(enabled: bool):
    torch._inductor.config._micro_pipeline_tp = enabled  # type: ignore


def run_warmup_batches(
    config: ServerConfig,
    input_q: mp.Queue,
    process_name: str,
    preprocess,
    run_model,
    synchronize,
    postprocess,
    device: DeviceType,
    dtype: torch.dtype,
):
    """
    Send a max-sized batch to the model to check if it's gonna OOM.

    Can also send more batches to try to trigger recompiles ahead of time.
    """

    max_tokens_per_forward = math.ceil(config.max_tokens_per_forward / config.pp_size)
    max_decode_tokens_per_forward = min(
        config.max_seqs_per_forward, max_tokens_per_forward
    )

    configs = []

    decode_sizes = [
        0,
        1 * config.tp_size,
        2 * config.tp_size,
        max_decode_tokens_per_forward,
    ]

    prefill_sizes = [
        0,
        1 * config.tp_size,
        2 * config.tp_size,
        max_tokens_per_forward - max_decode_tokens_per_forward,
        max_tokens_per_forward,
    ]

    if config.async_tp_threshold is not None:
        prefill_sizes.extend(
            range(
                config.async_tp_threshold - 3,
                config.async_tp_threshold + 1,
            )
        )

    for num_decode_tokens in decode_sizes:
        for num_prefill_tokens in prefill_sizes:
            total_tokens = num_prefill_tokens + num_decode_tokens
            if total_tokens <= 0 or total_tokens > max_tokens_per_forward:
                continue

            configs.append((num_prefill_tokens, num_decode_tokens))

    # sort configs by biggest first (and then tie-break prioritizing decode),
    # so we can discover OOMs as soon as possible
    configs.sort(key=lambda x: (x[0] + x[1], x[1]), reverse=True)

    inputs: list[ModelInput] = []

    for num_prefill_tokens, num_decode_tokens in configs:
        inp = make_dummy_batch(
            config,
            num_prefill_tokens,
            num_decode_tokens,
            skip_pipeline_communication=True,
        )
        inputs.append(inp)

    if config.pp_size > 1:
        for inp in inputs.copy():
            copy_inp = deepcopy(inp)
            copy_inp.skip_pipeline_communication = False
            inputs.append(copy_inp)

    for inp in inputs:
        input_q.put(inp)

    input_q.put(NoMoreInputs())

    run_overlapped_loop(
        preprocess,
        run_model,
        synchronize,
        postprocess,
        max_iters=len(inputs),
        # prog_bar_name=f"Warmup loop for {process_name}",
    )

    # Triggering the compilation/loading of the flashinfer merge
    # kernel here during server startup.
    # TODO: actually send hydragen batches in the warmup loop instead of this.
    if config.use_hydragen:
        bs = 8
        num_heads = 4
        hdim = 256
        out1 = torch.randn(bs, num_heads, hdim, device=device, dtype=dtype)
        out2 = torch.randn(bs, num_heads, hdim, device=device, dtype=dtype)
        lse1 = torch.randn(bs, num_heads, device=device, dtype=torch.float32)
        lse2 = torch.randn(bs, num_heads, device=device, dtype=torch.float32)

        cascade.merge_state(
            out1,
            lse1,
            out2,
            lse2,
        )


@dataclass
class CUDAGraphInfo:
    config: ServerConfig
    graph: torch.cuda.CUDAGraph
    input_batch_state: BatchState
    output_batch_state: BatchState
    wrappers: WrapperCollection
    model: nn.Module
    num_decode_tokens: int

    def __post_init__(self):
        self.pp_rank = self.model.extra_config.pp_rank
        self.pp_size = self.model.extra_config.pp_size

    def copy_into_input_batch_state(
        self, new_input_batch_state: BatchState, non_blocking: bool = False
    ):
        pp_rank = self.pp_rank
        pp_size = self.pp_size

        def copy_into(src: Tensor | None, dst: Tensor | None):
            assert src is not None
            assert dst is not None

            if src.shape[0] > dst.shape[0]:
                raise ValueError(
                    f"src.shape[0]={src.shape[0]} > dst.shape[0]={dst.shape[0]}"
                )
            dst[: src.shape[0]].copy_(src, non_blocking=non_blocking)

        copy_into(
            new_input_batch_state.position_ids,
            self.input_batch_state.position_ids,
        )

        # this is the only tensor that actually affects the state of our model
        # we point unused indices (i.e. padding) to the dummy last page we've added
        # to the graph to avoid overwriting stuff we care about.
        self.input_batch_state.attention_info.append_kv_token_indices.fill_(
            self.config.kv_cache_num_blocks() * self.config.page_size
        )
        copy_into(
            new_input_batch_state.attention_info.append_kv_token_indices,
            self.input_batch_state.attention_info.append_kv_token_indices,
        )

        if pp_rank == 0:
            copy_into(new_input_batch_state.input_ids, self.input_batch_state.input_ids)

        if pp_rank == pp_size - 1:
            copy_into(
                new_input_batch_state.lm_head_indices,
                self.input_batch_state.lm_head_indices,
            )

            sampling_params = self.input_batch_state.sampling_params
            assert sampling_params is not None
            if (greedy_mask := sampling_params.greedy_mask) is not None:
                copy_into(
                    new_input_batch_state.sampling_params.greedy_mask, greedy_mask
                )

            if (temperature := sampling_params.temperature) is not None:
                copy_into(
                    new_input_batch_state.sampling_params.temperature, temperature
                )

            if (top_p := sampling_params.top_p) is not None:
                copy_into(new_input_batch_state.sampling_params.top_p, top_p)

        if pp_rank > 0:
            copy_into(
                new_input_batch_state.hidden_states,
                self.input_batch_state.hidden_states,
            )

    def run(
        self,
        input_batch_state: BatchState,
        non_blocking: bool = False,
    ):
        assert input_batch_state.attention_info.hydragen_info is None
        batch_size = input_batch_state.position_ids.shape[0]

        self.copy_into_input_batch_state(input_batch_state, non_blocking)
        self.graph.replay()

        if self.pp_rank == self.pp_size - 1:
            assert self.output_batch_state.output_ids is not None
            assert self.output_batch_state.logprobs is not None

            input_batch_state.output_ids = self.output_batch_state.output_ids[
                :batch_size
            ]
            input_batch_state.logprobs = self.output_batch_state.logprobs[:batch_size]

        assert self.output_batch_state.hidden_states is not None
        input_batch_state.hidden_states = self.output_batch_state.hidden_states[
            :batch_size
        ]

        return input_batch_state

    def plan(self, input_batch_state: BatchState, non_blocking: bool = False):
        assert input_batch_state.attention_info.hydragen_info is None
        self.model.set_wrappers(self.wrappers)
        self.model.plan(input_batch_state.attention_info, non_blocking=non_blocking)


@torch.inference_mode()
def create_cudagraph(
    config: ServerConfig,
    model: nn.Module,
    num_decode_tokens: int,
    pp_rank: int,
    tp_rank: int,
    workspace_buffer: Tensor | None = None,
):
    tp_size = config.tp_size
    assert num_decode_tokens % tp_size == 0

    device = model.device
    orig_wrappers = model.wrapper_collection

    dummy_inp = make_dummy_batch(
        config=config,
        prefill_tokens=0,
        decode_tokens=num_decode_tokens,
    )

    input_batch_state = make_input_batch_state(
        inp=dummy_inp,
        tp_rank=tp_rank,
        tp_size=config.tp_size,
        pp_rank=pp_rank,
        pp_size=config.pp_size,
    )

    if pp_rank == 0:
        add_decoding_ids_to_batch_state(
            input_batch_state=input_batch_state,
            decoding_input_ids=torch.zeros(
                num_decode_tokens,
                dtype=torch.long,
            ),
            tp_rank=tp_rank,
            tp_size=config.tp_size,
        )

    move_batch_state(
        input_batch_state,
        device=device,
    )

    if pp_rank != 0:
        input_batch_state.hidden_states = torch.zeros(
            input_batch_state.position_ids.shape[0] // tp_size,
            model.config.hidden_size,
            dtype=model.dtype,
            device=device,
        )

    wrappers = create_wrappers_for_cudagraph(
        device=model.device,
        num_attention_heads=model.num_qo_heads(),
        num_key_value_heads=model.num_kv_heads(),
        num_decode_sequences=num_decode_tokens,
        max_kv_indices=config.cudagraph_max_kv_indices_per_seq * num_decode_tokens,
        workspace_buffer=workspace_buffer,
    )

    model.set_wrappers(wrappers)
    model.plan(input_batch_state.attention_info)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # type: ignore
        # warmup
        for _ in range(3):
            model(input_batch_state)

    torch.cuda.current_stream().wait_stream(s)

    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_batch_state = model(input_batch_state)

    model.set_wrappers(orig_wrappers)

    return CUDAGraphInfo(
        config=config,
        graph=g,
        input_batch_state=input_batch_state,
        output_batch_state=out_batch_state,
        wrappers=wrappers,
        model=model,
        num_decode_tokens=num_decode_tokens,
    )


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


class ModelRunner:
    def __init__(
        self,
        config: ServerConfig,
        model: nn.Module,
        process_group: dist.ProcessGroup,
        q_server_to_model: mp.Queue,
        device: str | torch.device,
    ):
        self.config = config
        self.model = model
        self.process_group = process_group
        self.default_wrappers = model.wrapper_collection
        self.q_server_to_model = q_server_to_model
        self.device = device
        assert self.default_wrappers is not None

        self.recorded_graphs = False

    def initialize(self):
        self.model.to_empty(device=self.device)

    def cleanup(self):
        self.model.to("meta")
        clean_memory()

    def should_use_async_tp_model(self, batch_state: BatchState):
        num_tokens = batch_state.position_ids.shape[0]
        return (
            self.config.async_tp_threshold is not None
            and num_tokens >= self.config.async_tp_threshold
        )

    def update_state_dict(self, command: UpdateStateDict):
        state_dict = self.model.state_dict()
        keys = sorted(state_dict.keys())

        works = []

        for key in keys:
            tensor = state_dict[key]

            if command.method == "distributed":
                if tensor.is_meta:
                    tensor = torch.empty_like(tensor, device=self.device)
                    state_dict[key] = tensor

                dist_work = dist.broadcast(
                    tensor, group=self.process_group, group_src=0, async_op=True
                )
                works.append(dist_work)

            else:
                tensor = self.q_server_to_model.get()
                state_dict[key] = tensor.clone()
                del tensor

        [work.wait() for work in works]

        self.model.load_state_dict(state_dict, assign=True)

    def run_default(
        self,
        input_batch_state: BatchState,
    ):
        use_async_tp = self.should_use_async_tp_model(input_batch_state)

        self.model.set_wrappers(self.default_wrappers)

        set_async_tp_enabled(use_async_tp)
        output_batch_state: BatchState = self.model(
            input_batch_state, async_tp=use_async_tp
        )
        set_async_tp_enabled(False)

        return output_batch_state

    def plan_default(
        self,
        input_batch_state: BatchState,
        non_blocking: bool = False,
    ):
        self.model.set_wrappers(self.default_wrappers)
        self.model.plan(input_batch_state.attention_info, non_blocking=non_blocking)

    def match_to_graph(self, input_batch_state: BatchState):
        num_prefill_tokens = input_batch_state.attention_info.prefill_info.num_tokens
        num_decode_tokens = input_batch_state.attention_info.decode_info.num_tokens

        if (
            num_prefill_tokens > 0
            or num_decode_tokens > self.config.cudagraph_max_size
            or not self.recorded_graphs
        ):
            return None

        assert num_decode_tokens > 0

        for i, graph in enumerate(self.graphs):
            graph_tokens = graph.num_decode_tokens
            if num_decode_tokens <= graph_tokens:
                return i

        raise RuntimeError(f"Shouldn't get here, num_decode_tokens={num_decode_tokens}")

    @torch.inference_mode()
    def plan(self, input_batch_state: BatchState, non_blocking: bool = False):
        graph_index = self.match_to_graph(input_batch_state)
        if graph_index is None:
            self.plan_default(input_batch_state, non_blocking)
            return

        graph = self.graphs[graph_index]
        input_batch_state.attention_info.decode_info.pad_for_cudagraph(
            graph.num_decode_tokens
        )
        graph.plan(input_batch_state, non_blocking)

    @torch.inference_mode()
    def run(
        self,
        input_batch_state: BatchState,
        non_blocking: bool = False,
    ):
        graph_index = self.match_to_graph(input_batch_state)
        if graph_index is None:
            return self.run_default(input_batch_state)

        return self.graphs[graph_index].run(input_batch_state, non_blocking)

    def record_graphs(self, process_name: str):
        # reuse the workspace buffer from the default wrappers
        workspace_buffer = (
            self.model.wrapper_collection.decode_wrapper._float_workspace_buffer
        )

        max_bs = self.config.cudagraph_max_size
        step = self.config.cudagraph_step

        assert max_bs % step == 0

        cuda_graph_sizes = list(range(step, max_bs + 1, step))

        graphs = list[CUDAGraphInfo]()
        for num_decode_tokens in tqdm(
            cuda_graph_sizes,
            desc=f"Capturing cudagraphs for {process_name}",
            disable=len(cuda_graph_sizes) == 0,
        ):
            graphs.append(
                create_cudagraph(
                    config=self.config,
                    model=self.model,
                    num_decode_tokens=num_decode_tokens,
                    pp_rank=self.model.extra_config.pp_rank,
                    tp_rank=self.model.extra_config.tp_rank,
                    workspace_buffer=workspace_buffer,
                )
            )

        self.graphs = graphs
        self.recorded_graphs = True


def setup_and_run_loop(
    state: BasicWorkerState | PipelineWorkerState,
    model_runner: ModelRunner,
    preprocess,
    run_model,
    synchronize,
    postprocess,
):
    if state.config.use_cudagraphs:
        model_runner.record_graphs(state.process_name)

    run_warmup_batches(
        config=state.config,
        input_q=state.input_q,
        process_name=state.process_name,
        preprocess=preprocess,
        run_model=run_model,
        synchronize=synchronize,
        postprocess=lambda _: None,
        device=model_runner.model.device,
        dtype=model_runner.model.dtype,
    )

    state.barrier.wait()

    run_overlapped_loop(
        preprocess=preprocess,
        run_model=run_model,
        synchronize=synchronize,
        postprocess=postprocess,
    )


def unpad_output_batch_state(
    output_batch_state: BatchState,
    input_batch_state: BatchState,
):
    if (lm_padding := input_batch_state.lm_head_indices_padding) > 0:
        assert output_batch_state.output_ids is not None
        assert output_batch_state.logprobs is not None
        output_batch_state.output_ids = output_batch_state.output_ids[:-lm_padding]
        output_batch_state.logprobs = output_batch_state.logprobs[:-lm_padding]


@contextmanager
def timer(
    name: str, enable: bool = True, min_ms: float | None = None, profile: bool = False
):
    if not enable:
        yield
        return

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    start = time.time()

    yield

    end = time.time()

    if profile:
        profiler.disable()

    ms = (end - start) * 1000
    if min_ms is None or ms > min_ms:
        print(f"timer {name}: {ms:.2f}ms")
        if profile:
            profiler.print_stats()


@contextmanager
def profile(name: str):
    """
    use built in python profiler
    """
    profiler = cProfile.Profile()
    profiler.enable()
    yield
    profiler.disable()
    print(f"Profiling {name}:")
    profiler.print_stats()


def get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def set_rank(rank: int):
    os.environ["LOCAL_RANK"] = str(rank)


def set_world_size(world_size: int):
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)


def set_master_port(port: int):
    os.environ["MASTER_PORT"] = str(port)


def get_master_port() -> int:
    return int(os.environ.get("MASTER_PORT", "29500"))


def set_master_addr(addr: str):
    os.environ["MASTER_ADDR"] = addr


def get_master_addr() -> str:
    return os.environ.get("MASTER_ADDR", "localhost")


def is_local() -> bool:
    return get_rank() == 0


def lprint(*args, **kwargs):
    if is_local():
        print(*args, **kwargs)


def ltqdm(*args, **kwargs):
    if is_local():
        return tqdm(*args, **kwargs)
    else:
        return tqdm(*args, **kwargs, disable=True)


def lprint_tensor(tensor: torch.Tensor):
    lprint(tensor.sum(), tensor.view(-1)[:5], tensor.view(-1)[-5:])


def std(lst):
    mean = sum(lst) / len(lst)
    return (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5


def median(lst):
    sorted_lst = sorted(lst)
    if len(sorted_lst) % 2 == 1:
        return sorted_lst[len(sorted_lst) // 2]
    else:
        return (
            sorted_lst[len(sorted_lst) // 2 - 1] + sorted_lst[len(sorted_lst) // 2]
        ) / 2


@dataclass
class TimeResult:
    times: list[float]
    warmup_times: list[float]
    cpu_times: list[float]
    cpu_warmup_times: list[float]

    def mean(self):
        return sum(self.times) / len(self.times)

    def std(self):
        return std(self.times)

    def cpu_mean(self):
        return sum(self.cpu_times) / len(self.cpu_times)

    def cpu_std(self):
        return std(self.cpu_times)

    def median(self):
        return median(self.times)

    def cpu_median(self):
        return median(self.cpu_times)


def convert_unit(milis: float, unit: str) -> float:
    if unit == "ms":
        return milis
    elif unit == "s":
        return milis / 1000
    elif unit == "us":
        return milis * 1000
    else:
        raise ValueError(f"Invalid unit: {unit}")


@torch.no_grad()
def timed(
    fn,
    num_iters=50,
    num_warmup=10,
    unit: str = "ms",
    between_fn=None,
    prog=True,
    barrier=False,
):
    warmup_times = []
    times = []
    cpu_warmup_times = []
    cpu_times = []
    for itr in tqdm(range(num_iters + num_warmup), desc="Timing", disable=not prog):
        if between_fn is not None:
            between_fn()
            torch.cuda.synchronize()

        if barrier:
            torch.distributed.barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()  # type: ignore
        cpu_start = time.perf_counter()
        _ = fn()
        cpu_end = time.perf_counter()
        end.record()  # type: ignore
        torch.cuda.synchronize()

        if barrier:
            torch.distributed.barrier()

        gpu_milis = start.elapsed_time(end)
        cpu_milis = (cpu_end - cpu_start) * 1000

        gpu_time = convert_unit(gpu_milis, unit)
        cpu_time = convert_unit(cpu_milis, unit)

        if itr >= num_warmup:
            times.append(gpu_time)
            cpu_times.append(cpu_time)
        else:
            warmup_times.append(gpu_time)
            cpu_warmup_times.append(cpu_time)

    return TimeResult(
        times=times,
        warmup_times=warmup_times,
        cpu_times=cpu_times,
        cpu_warmup_times=cpu_warmup_times,
    )


@torch.no_grad()
def timed_with_graph(
    fn,
    num_iters=50,
    num_warmup=10,
    unit: str = "ms",
    prog=True,
):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # type: ignore
        # warmup
        for _ in ltqdm(range(3)):
            fn()

    torch.cuda.current_stream().wait_stream(s)

    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    warmup_times = []
    times = []
    cpu_warmup_times = []
    cpu_times = []
    for itr in tqdm(range(num_iters + num_warmup), desc="Timing", disable=not prog):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cpu_start = time.perf_counter()
        g.replay()
        cpu_end = time.perf_counter()
        end.record()
        torch.cuda.synchronize()

        gpu_milis = start.elapsed_time(end)
        cpu_milis = (cpu_end - cpu_start) * 1000

        gpu_time = convert_unit(gpu_milis, unit)
        cpu_time = convert_unit(cpu_milis, unit)

        if itr >= num_warmup:
            times.append(gpu_time)
            cpu_times.append(cpu_time)
        else:
            warmup_times.append(gpu_time)
            cpu_warmup_times.append(cpu_time)

    return TimeResult(
        times=times,
        warmup_times=warmup_times,
        cpu_times=cpu_times,
        cpu_warmup_times=cpu_warmup_times,
    )


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def terminate_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        for child in children:
            child.terminate()

        gone, alive = psutil.wait_procs(children, timeout=5)

        for p in alive:
            p.kill()

        parent.terminate()
        parent.wait(5)
    except psutil.NoSuchProcess:
        pass


@dataclass
class ServerArgs:
    args: list[str]
    start_server: bool = True
    port: int | None = None


@contextmanager
def sglang_manager(config: ServerArgs):
    if not config.start_server:
        yield None
        return config.port

    if config.args is None:
        args = ""
    else:
        str_args = [str(arg) for arg in config.args]
        args = " ".join(str_args)

    port = find_free_port()

    launch_command = f"""python -m sglang.launch_server \
        --port {port} \
        {args}"""

    print(f"Starting sglang server with command: {launch_command}")
    server_process = subprocess.Popen(launch_command, shell=True)
    print(f"Started sglang server with pid {server_process.pid}")

    try:
        wait_for_ping(port, server_process, max_retries=500, ping_endpoint="health")
        yield port
    finally:
        print(f"Killing sglang server (pid {server_process.pid})...")
        terminate_process_tree(server_process.pid)
        print("Done killing sglang server.")


@contextmanager
def vllm_manager(config: ServerArgs):
    if not config.start_server:
        yield None
        return config.port

    if config.args is None:
        args = ""
    else:
        str_args = [str(arg) for arg in config.args]
        args = " ".join(str_args)

    port = find_free_port()

    vllm_command = f"""python vllm_server.py \
        --port {port} \
        {args}"""

    print(f"Starting vllm server with command: {vllm_command}")
    vllm_process = subprocess.Popen(vllm_command, shell=True)
    print(f"Started vllm server with pid {vllm_process.pid}")

    try:
        wait_for_ping(port, vllm_process, max_retries=500)
        yield port
    finally:
        print(f"Killing vllm server (pid {vllm_process.pid})...")
        terminate_process_tree(vllm_process.pid)
        print("Done killing vllm server.")


def wait_for_ping(
    port,
    popen: subprocess.Popen,
    retry_seconds=2,
    max_retries=500,
    ping_endpoint: str = "ping",
):
    # wait for the server to start, by /ping-ing it
    print(f"Waiting for server to start on port {port}...")
    for i in range(max_retries):
        try:
            requests.get(f"http://localhost:{port}/{ping_endpoint}")
            return
        except requests.exceptions.ConnectionError:
            if popen.poll() is not None:
                raise RuntimeError(
                    f"Server died with code {popen.returncode} before starting."
                )

            print(f"Server not yet started (attempt {i}) retrying...")
            time.sleep(retry_seconds)

    raise RuntimeError(f"Server not started after {max_retries} attempts.")


def gpus_to_cvd(gpus: list[int]):
    return "CUDA_VISIBLE_DEVICES=" + ",".join([str(x) for x in gpus])


def save_pkl(obj, path: str):
    """Save an object to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path: str):
    """Load an object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_eos_token_ids(generation_config):
    model_eos = generation_config.eos_token_id
    if model_eos is None:
        eos_token_ids = []
    elif isinstance(model_eos, int):
        eos_token_ids = [model_eos]
    else:
        assert isinstance(model_eos, list)
        eos_token_ids = model_eos

    return set(eos_token_ids)


def setup_logging(config: "ServerConfig"):
    pass


def queue_iterator(q: mp.Queue):
    while True:
        try:
            yield q.get_nowait()
        except queue.Empty:
            break


def block_on_queues(queues: list[mp.Queue]):
    readers = [q._reader for q in queues]
    mp_conn.wait(readers, timeout=None)


def error_propogation_decorator(func):
    """
    Sometimes, for weird reasons, error messages are
    silenced in engine subprocesses. Here we explicitly
    print to stdout to ensure the error is visible.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print("Caught error - reprinting and reraising")
            print("-" * 80)
            print(e)
            traceback.print_exc()
            print("-" * 80)
            raise

    return wrapper
