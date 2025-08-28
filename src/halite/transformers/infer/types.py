from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from halite.transformers.infer.engine.batch import Batch, ForwardBatch


class InferenceRequest(NamedTuple):
    id: str
    input_ids: list[int]
    sampling_params: dict[str, Any]


class InferenceResult:
    def __init__(
        self,
        id: int | str,
        input_ids: list[int] | None = None,
        output_ids: list[list[int]] | None = None,
        output_logprobs: list[list[float]] | None = None,
        finish_reason: list[str] | None = None,
        num_cached_prompt_tokens: list[int] | None = None,
    ):
        self.id = id

        self.input_ids = [] if input_ids is None else input_ids
        self.output_ids = [] if output_ids is None else output_ids

        self.output_logprobs = [] if output_logprobs is None else output_logprobs
        self.finish_reason = [] if finish_reason is None else finish_reason
        self.num_cached_prompt_tokens = (
            [] if num_cached_prompt_tokens is None else num_cached_prompt_tokens
        )

    def validate_lengths(self):
        if not (
            len(self.output_ids)
            == len(self.output_logprobs)
            == len(self.finish_reason)
            == len(self.num_cached_prompt_tokens)
        ):
            raise ValueError(
                f"lengths of output_ids, logprobs, finish_reason, and num_cached_prompt_tokens must be equal. Got {len(self.output_ids)}, {len(self.logprobs)}, {len(self.finish_reason)}, and {len(self.num_cached_prompt_tokens)}"
            )

    def to_dict(self):
        return {
            "id": self.id,
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
            "output_logprobs": self.output_logprobs,
        }

    def __repr__(self):
        if len(self.output_ids) > 2:
            output_ids = [self.output_ids[0], "...", self.output_ids[-1]]

        else:
            output_ids = self.output_ids

        output_ids = ", ".join(str(out) for out in output_ids)

        return f"InferenceResult(id={self.id}, input_ids={self.input_ids}, output_ids=[{output_ids}], output_logprobs={self.output_logprobs})"


@dataclass
class ServerConfig:
    max_prefill_tokens: int = 16384
    chunked_prefill_size: int = 8192

    default_init_new_token_ratio: float = 0.7
    default_min_new_token_ratio_factor: float = 0.14
    default_new_token_ratio_decay_steps: int = 600
    schedule_conservativeness: float = 1.0

    use_cudagraph: bool = False

    cudagraph_step: int = 16
    cudagraph_additonal_batch_size: tuple[int] = ()
    cudagraph_max_batch_size: int = 512


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


class CUDAGraphState:
    def __init__(
        self,
        forward_batch: ForwardBatch,
        request_pool_ids: torch.Tensor,
    ):
        self.forward_batch = forward_batch
        self.request_pool_ids = request_pool_ids

    def max_batch_size(self):
        return self.forward_batch.input_ids.shape[0]

    def get_forward_batch(self, batch_size: int):
        return ForwardBatch(
            self.forward_batch.input_ids[:batch_size],
            self.forward_batch.kv_pool_ids[:batch_size],
            self.forward_batch.seq_lens[:batch_size],
            self.forward_batch.extend_lens[:batch_size],
            self.forward_batch.positions[:batch_size],
            self.forward_batch.mode,
        )

    def get_request_pool_ids(self, batch_size: int):
        return self.request_pool_ids[:batch_size]

    def set_request_pool_ids(self, request_pool_ids: torch.Tensor):
        self.request_pool_ids[: request_pool_ids.shape[0]] = request_pool_ids

    def set_forward_batch(self, forward_batch: Batch | ForwardBatch):
        batch_size = forward_batch.input_ids.shape[0]

        self.forward_batch.input_ids[:batch_size] = forward_batch.input_ids
        self.forward_batch.kv_pool_ids[:batch_size] = forward_batch.kv_pool_ids
        self.forward_batch.seq_lens[:batch_size] = forward_batch.seq_lens
        # self.forward_batch.extend_lens[:batch_size] = forward_batch.extend_lens
        self.forward_batch.positions[:batch_size] = (forward_batch.seq_lens - 1).to(
            torch.int64
        )
        self.forward_batch.mode = forward_batch.mode
