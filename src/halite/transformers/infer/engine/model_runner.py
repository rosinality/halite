import torch

from halite.transformers.infer.sampler import Sampler
from halite.transformers.infer.memory_pool import RequestToTokenPool, MHAKVPool
from halite.transformers.infer.device import get_available_gpu_memory
from halite.transformers.infer.flashinfer_backend import FlashInferBackend


class ModelRunner:
    def __init__(
        self,
        model,
        n_head,
        n_key_value_head,
        head_dim,
        n_layer,
        context_len,
        memory_fraction_static,
        kv_cache_dtype,
        gpu_id,
        device,
        distributed=False,
    ):
        self.model = model
        self.sampler = Sampler()

        self.n_kv_head = n_key_value_head
        self.head_dim = head_dim
        self.n_layer = n_layer
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
            n_head=n_key_value_head,
            head_dim=head_dim,
            n_layer=n_layer,
            device=device,
        )
        self.attention_backend = FlashInferBackend(
            n_head,
            n_key_value_head,
            head_dim,
            is_causal=True,
            normalize=None,
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
