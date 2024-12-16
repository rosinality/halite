from dataclasses import dataclass

import torch

from halite.transformers.infer.engine.sampler import Sampler
from halite.transformers.infer.engine.memory_pool import RequestToTokenPool, MHAKVPool
from halite.transformers.infer.engine.device import get_available_gpu_memory
from halite.transformers.infer.engine.flashinfer_backend import FlashInferBackend


@dataclass
class ModelConfig:
    n_heads: int
    n_key_value_heads: int
    head_dim: int
    n_layers: int
    context_len: int
    tp_size: int = 1
    memory_fraction_static: float | None = None
    kv_cache_dtype: torch.dtype = "auto"
    gpu_id: int = 0
    device: torch.device | str = "cuda"
    distributed: bool = False

    def __post_init__(self):
        if self.memory_fraction_static is None:
            if self.tp_size >= 16:
                self.memory_fraction_static = 0.79

            elif self.tp_size >= 8:
                self.memory_fraction_static = 0.81

            elif self.tp_size >= 4:
                self.memory_fraction_static = 0.85

            elif self.tp_size >= 2:
                self.memory_fraction_static = 0.87

            else:
                self.memory_fraction_static = 0.88


class ModelRunner:
    def __init__(
        self,
        model,
        model_config: ModelConfig,
    ):
        self.model = model
        self.sampler = Sampler()

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

        self.memory_fraction_static = memory_fraction_static
        self.gpu_id = gpu_id
        self.device = device
        self.distributed = distributed

        min_per_gpu_memory = get_available_gpu_memory(
            device, gpu_id, distributed=distributed
        )
        self.max_total_tokens = self.estimate_max_n_tokens(min_per_gpu_memory)
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

    def estimate_max_n_tokens(self, gpu_memory: int):
        available_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.distributed
        )

        cell_size = (
            self.n_kv_head
            * self.head_dim
            * self.n_layer
            * 2
            * torch._utils._element_size(self.kv_cache_dtype)
        )

        rest_memory = available_memory - gpu_memory * (1 - self.memory_fraction_static)
        max_n_tokens = int(rest_memory * (1 << 30) // cell_size)

        return max_n_tokens

    def forward(self, batch):
        batch.prepare_forward(self.attention_backend)

        return self.model(batch)

    def sample(self, logits, batch):
        sampling_params = batch.sampling_params
        next_token_ids = self.sampler.forward(logits, sampling_params)

        return next_token_ids

    def forward_and_sample(self, batch):
        logits_output = self.forward(batch)
        next_token_ids = self.sample(logits_output, batch)

        return logits_output, next_token_ids
