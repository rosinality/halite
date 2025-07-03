import bisect

import torch

from halite.distributed.state_dict import reshard_state_dict
from halite.transformers.infer.engine.batch import ForwardBatch, ForwardMode
from halite.transformers.infer.engine.sampler import Sampler
from halite.transformers.infer.engine.memory_pool import RequestToTokenPool, MHAKVPool
from halite.transformers.infer.engine.device import get_gpu_memory
from halite.transformers.infer.engine.flashinfer_backend import FlashInferBackend
from halite.transformers.infer.types import ModelConfig, ServerConfig, CUDAGraphState


class ModelRunner:
    def __init__(
        self,
        model,
        model_config: ModelConfig,
        server_config: ServerConfig,
    ):
        self.model = model
        self.sampler = Sampler()
        self.server_config = server_config

        n_heads = model_config.n_heads
        n_key_value_heads = model_config.n_key_value_heads
        head_dim = model_config.head_dim
        n_layers = model_config.n_layers
        context_len = model_config.context_len
        memory_fraction_static = model_config.memory_fraction_static
        kv_cache_dtype = model_config.kv_cache_dtype
        gpu_id = model_config.gpu_id
        device = model_config.device
        distributed = model_config.distributed

        if kv_cache_dtype == "auto":
            kv_cache_dtype = model.dtype

        self.n_kv_head = n_key_value_heads
        self.head_dim = head_dim
        self.n_layer = n_layers
        self.kv_cache_dtype = kv_cache_dtype
        self.context_len = context_len

        self.memory_fraction_static = memory_fraction_static
        self.gpu_id = gpu_id
        self.device = device
        self.distributed = distributed

        avail_memory, total_memory = get_gpu_memory(
            device, gpu_id, distributed=distributed
        )
        self.max_total_tokens = self.estimate_max_n_tokens(avail_memory, total_memory)
        self.max_running_requests = self.max_total_tokens // 2
        self.max_requests = min(
            max(int(self.max_total_tokens / context_len * 512), 2048), 4096
        )

        self.request_to_token_pool = RequestToTokenPool(
            max_size=self.max_requests, max_context_len=context_len + 4, device=device
        )
        self.kv_pool = MHAKVPool(
            self.max_total_tokens,
            dtype=kv_cache_dtype,
            n_heads=n_key_value_heads,
            head_dim=head_dim,
            n_layers=n_layers,
            device=device,
        )
        self.attention_backend = FlashInferBackend(
            n_heads,
            n_key_value_heads,
            head_dim,
            is_causal=True,
            scale=None,
            dtype=kv_cache_dtype,
            request_to_token=self.request_to_token_pool,
            max_batch_size=self.request_to_token_pool.max_size,
            max_context_len=context_len,
            device=device,
        )

        self.cudagraph_state = None
        self.cudagraphs = {}
        self.cudagraph_outputs = {}
        self.cudagraph_sizes = (
            self.server_config.cudagraph_additonal_batch_size
            + tuple(
                range(
                    self.server_config.cudagraph_step,
                    self.server_config.cudagraph_max_batch_size,
                    self.server_config.cudagraph_step,
                )
            )
        )

    def initialize(self):
        self.request_to_token_pool.initialize()
        self.kv_pool.initialize()
        self.attention_backend.initialize()

    def cleanup(self):
        self.model.to("meta")
        self.request_to_token_pool.cleanup()
        self.kv_pool.cleanup()
        self.attention_backend.cleanup()

    def load_state_dict(self, state_dict, assign=True):
        self.model.to("meta")
        state_dict = reshard_state_dict(state_dict, self.model.state_dict())

        if not assign:
            self.model.to_empty(device=self.device)

        self.model.load_state_dict(state_dict, assign=assign)

    def estimate_max_n_tokens(self, available_memory: int, total_memory: int):
        cell_size = (
            self.n_kv_head
            * self.head_dim
            * self.n_layer
            * 2
            * torch._utils._element_size(self.kv_cache_dtype)
        )

        rest_memory = available_memory - total_memory * (
            1 - self.memory_fraction_static
        )
        max_n_tokens = int(rest_memory * (1 << 30) // cell_size)

        return max_n_tokens

    def forward(self, batch):
        if (
            batch.mode == ForwardMode.DECODE
            and self.server_config.use_cudagraph
            and batch.input_ids.shape[0] <= self.server_config.cudagraph_max_batch_size
        ):
            return self.forward_cudagraph(batch)
        else:
            return self.forward_eager(batch)

    def forward_eager(self, batch):
        batch.prepare_forward(self.attention_backend)
        self.model.set_attention_backend(batch.attention_backend)
        self.model.set_kv_pool(batch.kv_pool)

        forward_batch = ForwardBatch(
            batch.input_ids,
            batch.kv_pool_ids,
            batch.seq_lens,
            batch.extend_lens,
            batch.positions,
            batch.mode,
        )

        return self.model(forward_batch)

    @torch.inference_mode()
    def forward_cudagraph(self, batch):
        batch_size = batch.input_ids.shape[0]
        graph_index = bisect.bisect_left(self.cudagraph_sizes, batch_size)
        graph_batch_size = self.cudagraph_sizes[graph_index]

        if graph_batch_size not in self.cudagraphs:
            self.capture_cudagraph(graph_batch_size)

        graph = self.cudagraphs[graph_batch_size]
        output = self.cudagraph_outputs[graph_batch_size]
        self.cudagraph_state.set_forward_batch(batch)
        self.cudagraph_state.set_request_pool_ids(batch.request_pool_ids)

        forward_batch = self.cudagraph_state.get_forward_batch(graph_batch_size)

        self.attention_backend.decode_update_for_cudagraph(
            graph_batch_size,
            self.cudagraph_state.get_request_pool_ids(graph_batch_size),
            forward_batch.seq_lens,
            forward_batch.seq_lens.sum().item(),
        )

        self.model.set_attention_backend(self.attention_backend)
        self.model.set_kv_pool(self.kv_pool)

        graph.replay()

        # output = output.slice(batch_size)

        return output.slice(batch_size)

    def prepare_cudagraph(self):
        max_batch_size = self.server_config.cudagraph_max_batch_size
        input_ids = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        kv_pool_ids = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        seq_lens = torch.full(
            (max_batch_size,), self.context_len, dtype=torch.int32, device=self.device
        )
        extend_lens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        positions = torch.zeros(max_batch_size, dtype=torch.int64, device=self.device)
        request_pool_ids = torch.zeros(
            max_batch_size, dtype=torch.int32, device=self.device
        )
        mode = ForwardMode.DECODE

        forward_batch = ForwardBatch(
            input_ids, kv_pool_ids, seq_lens, extend_lens, positions, mode
        )

        self.cudagraph_state = CUDAGraphState(forward_batch, request_pool_ids)

        self.attention_backend.initialize_cudagraph_buffers(max_batch_size)
        self.graph_pool = None

    @torch.inference_mode()
    def capture_cudagraph(self, batch_size: int):
        if self.cudagraph_state is None:
            self.prepare_cudagraph()
            self.cudagraphs = {}

        self.attention_backend.initialize_cudagraph(batch_size)
        self.attention_backend.decode_update_for_cudagraph(
            batch_size,
            self.cudagraph_state.request_pool_ids[:batch_size],
            self.cudagraph_state.forward_batch.seq_lens[:batch_size],
            self.cudagraph_state.forward_batch.seq_lens[:batch_size].sum().item(),
        )

        graph = torch.cuda.CUDAGraph()
        forward_batch = self.cudagraph_state.get_forward_batch(batch_size)
        self.model.set_attention_backend(self.attention_backend)
        self.model.set_kv_pool(self.kv_pool)
        self.model(forward_batch)
        with torch.cuda.graph(graph, self.graph_pool):
            output = self.model(forward_batch)
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.cudagraphs[batch_size] = graph
        self.cudagraph_outputs[batch_size] = output
        torch.cuda.synchronize()

    def sample(self, logits, batch):
        sampling_params = batch.sampling_params
        next_token_ids = self.sampler.forward(logits, sampling_params)

        return next_token_ids

    def forward_and_sample(self, batch):
        logits_output = self.forward(batch)
        next_token_ids = self.sample(logits_output, batch)

        return logits_output, next_token_ids
