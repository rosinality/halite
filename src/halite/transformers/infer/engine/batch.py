from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any

import torch

from halite.transformers.infer.engine.memory_pool import RequestToTokenPool, KVPool
from halite.transformers.infer.engine.radix_cache import RadixCache


class ForwardMode(IntEnum):
    EXTEND = auto()
    DECODE = auto()

    def is_decode(self):
        return self == ForwardMode.DECODE


class SamplingParams:
    def __init__(
        self,
        max_new_tokens: int = 128,
        min_new_tokens: int = 0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0,
        n: int = 1,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        ignore_eos: bool = False,
    ):
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.n = n

        self.stop_strs = stop
        if stop_token_ids is not None:
            self.stop_token_ids = set(stop_token_ids)

        else:
            self.stop_token_ids = None

        if self.top_k == -1:
            self.top_k = 1 << 30

        self.ignore_eos = ignore_eos

        self.normalized = False

    def normalize(self, tokenizer):
        if self.normalized:
            return

        if self.stop_strs is None:
            self.stop_strs = []
            self.stop_str_max_len = 0

        else:
            if isinstance(self.stop_strs, str):
                self.stop_strs = [self.stop_strs]

            stop_str_max_len = 0
            for stop_str in self.stop_strs:
                if tokenizer is not None:
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))

                else:
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))

            self.stop_str_max_len = stop_str_max_len

        self.normalized = True


@dataclass
class BatchSamplingParams:
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor

    is_all_greedy: bool
    need_min_p_sampling: bool

    @classmethod
    def from_requests(cls, requests, device):
        temperatures = (
            torch.tensor(
                [r.sampling_params.temperature for r in requests], dtype=torch.float32
            )
            .view(-1, 1)
            .to(device, non_blocking=True)
        )
        top_ps = torch.tensor(
            [r.sampling_params.top_p for r in requests], dtype=torch.float32
        ).to(device, non_blocking=True)
        top_ks = torch.tensor(
            [r.sampling_params.top_k for r in requests], dtype=torch.int32
        ).to(device, non_blocking=True)
        min_ps = torch.tensor(
            [r.sampling_params.min_p for r in requests], dtype=torch.float32
        ).to(device, non_blocking=True)

        return cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=top_ks.max().item() <= 1,
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in requests),
        )

    def __len__(self):
        return len(self.temperatures)

    def filter(self, unfinished_ids, new_ids):
        for item in ("temperatures", "top_ps", "top_ks", "min_ps"):
            value = getattr(self, item, None)
            if value is not None:
                setattr(self, item, value[new_ids])

    def merge(self, other):
        for item in ("temperatures", "top_ps", "top_ks", "min_ps"):
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            setattr(self, item, torch.cat((self_val, other_val)))

        self.is_all_greedy = self.is_all_greedy and other.is_all_greedy


class FinishReason:
    def __init__(self, reason, is_error=False, length=None, matched=None):
        self.reason = reason
        self.is_error = is_error
        self.length = length
        self.matched = matched


class Request:
    def __init__(self, id, input_ids, sampling_params, input_text=None):
        self.id = id
        self.input_text = input_text
        self.input_ids = input_ids
        self.sampling_params = sampling_params
        self.output_ids = []
        self.all_ids = None

        self.decoded_text = ""
        self.finished_reason = None

        self.request_pool_id = None
        self.prefix_ids = []
        self.extend_input_len = 0
        self.last_node = None

        self.is_retracted = False
        self.completion_tokens_without_jump_forward = 0

        self.return_logprob = True
        self.logprob_start_len = 0
        self.top_logprobs_num = 0
        self.normalized_prompt_logprob = None
        self.input_token_logprobs = None
        self.input_top_logprobs = None
        self.output_logprobs = []
        self.output_top_logprobs = []

    def normalize_sampling_params(self, tokenizer):
        self.sampling_params.normalize(tokenizer)

    def finished(self):
        return self.finished_reason is not None

    def check_finished(self, tokenizer=None):
        if self.finished():
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FinishReason(
                "length", length=self.sampling_params.max_new_tokens
            )

            return

        last_token_id = self.output_ids[-1]
        matched_eos = False

        if self.sampling_params.stop_token_ids:
            matched_eos = last_token_id in self.sampling_params.stop_token_ids

        if tokenizer is not None:
            matched_eos |= last_token_id == tokenizer.eos_id

            if tokenizer.additional_stop_token_ids:
                matched_eos |= last_token_id in tokenizer.additional_stop_token_ids

        if matched_eos and not self.sampling_params.ignore_eos:
            self.finished_reason = FinishReason("stop", matched=last_token_id)

        if len(self.sampling_params.stop_strs) > 0:
            tail_str = tokenizer.decode(
                self.output_ids[-(self.sampling_params.stop_str_max_len + 1) :]
            )

            for stop_str in self.sampling_params.stop_strs:
                if stop_str in tail_str or stop_str in self.decoded_text:
                    self.finished_reason = FinishReason("stop", matched=stop_str)

                    return

    def adjust_max_prefix_ids(self):
        self.all_ids = self.input_ids + self.output_ids
        input_len = len(self.all_ids)

        max_prefix_len = input_len - 1

        if self.sampling_params.max_new_tokens > 0:
            max_prefix_len = min(max_prefix_len, input_len - 1)

        if self.return_logprob:
            if self.normalized_prompt_logprob is None:
                max_prefix_len = min(max_prefix_len, input_len - 2)

            max_prefix_len = min(max_prefix_len, self.logprob_start_len)

        max_prefix_len = max(max_prefix_len, 0)

        return self.all_ids[:max_prefix_len]

    def prepare_next_input(self, tree_cache: RadixCache | None = None):
        self.all_ids = self.input_ids + self.output_ids

        if tree_cache is not None:
            self.prefix_ids, self.last_node = tree_cache.match(
                key=self.adjust_max_prefix_ids()
            )

        self.extend_input_len = len(self.all_ids) - len(self.prefix_ids)

    def reset_for_retract(self):
        self.prefix_ids = []
        self.last_node = None
        self.extend_input_len = 0
        self.is_retracted = True

        self.logprob_start_len = 10**9


BATCH_ID = 0


@dataclass
class Batch:
    requests: list[Request]
    request_to_token_pool: RequestToTokenPool
    kv_pool: KVPool

    mode: ForwardMode | None = None
    attention_backend: Any | None = None
    tree_cache: RadixCache | None = None

    input_ids: torch.Tensor | None = None
    request_pool_ids: torch.Tensor | None = None
    kv_pool_ids: torch.Tensor | None = None
    seq_lens: list[int] | None = None
    output_ids: torch.Tensor | None = None
    prefix_lens: list[int] | None = None
    extend_lens: list[int] | None = None
    decoding_requests: list[Request] | None = None
    positions: torch.Tensor | None = None

    return_logprob: bool = False
    top_logprobs_nums: list[int] | None = None

    sampling_params: BatchSamplingParams | None = None

    device: str = "cuda"

    def __post_init__(self):
        global BATCH_ID

        self.id = BATCH_ID
        BATCH_ID += 1

        self.retract_decode_steps = 20
        self.return_logprob = any(req.return_logprob for req in self.requests)

    def alloc_request_slots(self, n_requests: int):
        req_pool_ids = self.request_to_token_pool.alloc(n_requests)

        return req_pool_ids

    def alloc_kv_slots(self, n_tokens: int):
        kv_pool_ids = self.kv_pool.alloc(n_tokens)

        if kv_pool_ids is None:
            if self.tree_cache is not None:
                self.tree_cache.evict(n_tokens, self.kv_pool.free)
                kv_pool_ids = self.kv_pool.alloc(n_tokens)

        return kv_pool_ids

    def batch_size(self):
        return len(self.requests)

    def is_empty(self):
        return len(self.requests) == 0

    def prepare_for_extend(self):
        self.mode = ForwardMode.EXTEND

        batch_size = len(self.requests)
        input_ids = [req.all_ids[len(req.prefix_ids) :] for req in self.requests]
        n_tokens = sum(len(ids) for ids in input_ids)

        req_pool_ids = self.alloc_request_slots(batch_size)
        kv_pool_ids = self.alloc_kv_slots(n_tokens)

        start = 0
        seq_lens = []
        for i, req in enumerate(self.requests):
            req.request_pool_id = req_pool_ids[i]
            prefix_len = len(req.prefix_ids)
            seq_len = len(req.all_ids)
            seq_lens.append(seq_len)

            assert seq_len - prefix_len == req.extend_input_len

            if prefix_len > 0:
                self.request_to_token_pool.write(
                    (req.request_pool_id, slice(0, prefix_len)), req.prefix_ids
                )

            self.request_to_token_pool.write(
                (req.request_pool_id, slice(prefix_len, seq_len)),
                kv_pool_ids[start : start + req.extend_input_len],
            )

            start += req.extend_input_len
            req.is_retracted = False

        self.input_ids = torch.tensor(sum(input_ids, []), dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        self.request_pool_ids = torch.tensor(req_pool_ids, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        self.kv_pool_ids = kv_pool_ids
        self.seq_lens_sum = sum(seq_lens)

        # if self.return_logprob:
        #     self.top_logprobs_nums = [req.top_logprobs_num for req in self.requests]

        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        self.prefix_lens = torch.tensor(
            [len(req.prefix_ids) for req in self.requests], dtype=torch.int32
        ).to(self.device, non_blocking=True)
        self.extend_lens = torch.tensor(
            [req.extend_input_len for req in self.requests], dtype=torch.int32
        ).to(self.device, non_blocking=True)
        self.extend_n_tokens = n_tokens

        self.sampling_params = BatchSamplingParams.from_requests(
            self.requests, self.device
        )

    def prepare_for_decode(self):
        self.mode = ForwardMode.DECODE

        self.input_ids = self.output_ids
        self.output_ids = None

        batch_size = len(self.requests)
        self.kv_pool_ids = self.alloc_kv_slots(batch_size)

        ids = self.seq_lens

        self.request_to_token_pool.write((self.request_pool_ids, ids), self.kv_pool_ids)
        self.seq_lens.add_(1)

        self.seq_lens_sum += batch_size

    def prepare_forward(self, attention_backend: Any):
        if self.mode.is_decode():
            self.positions = (self.seq_lens - 1).to(torch.int64)

        else:
            self.positions = torch.cat(
                [
                    torch.arange(
                        prefix_len, prefix_len + extend_len, device=self.device
                    )
                    for prefix_len, extend_len in zip(
                        self.prefix_lens, self.extend_lens
                    )
                ]
            )

        self.attention_backend = attention_backend
        self.attention_backend.prepare(self)

    def filter(self, keep_ids=None):
        if keep_ids is None:
            keep_ids = [i for i, req in enumerate(self.requests) if not req.finished()]

        if len(keep_ids) == 0:
            self.requests = []

            return

        if len(keep_ids) == len(self.requests):
            return

        self.requests = [self.requests[i] for i in keep_ids]
        new_ids = torch.tensor(keep_ids, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        self.request_pool_ids = self.request_pool_ids[new_ids]
        self.seq_lens = self.seq_lens[new_ids]
        self.kv_pool_ids = None
        self.seq_lens_sum = self.seq_lens.sum().item()
        self.output_ids = self.output_ids[new_ids]
        self.return_logprob = any(req.return_logprob for req in self.requests)
        # if self.return_logprob:
        #     self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_ids]
        # else:
        #     self.top_logprobs_nums = None
        self.top_logprobs_nums = None

        self.sampling_params.filter(keep_ids, new_ids)

    def merge(self, other):
        self.sampling_params.merge(other.sampling_params)

        self.request_pool_ids = torch.cat(
            (self.request_pool_ids, other.request_pool_ids)
        )
        self.seq_lens = torch.cat((self.seq_lens, other.seq_lens))
        self.seq_lens_sum += other.seq_lens_sum
        self.kv_pool_ids = None

        if self.output_ids is not None:
            self.output_ids = torch.cat((self.output_ids, other.output_ids))

        self.requests.extend(other.requests)

    def check_decode_memory(self):
        batch_size = len(self.requests)

        if self.kv_pool.available_size() >= batch_size:
            return True

        self.tree_cache.evict(batch_size, self.kv_pool.free)

        if self.kv_pool.available_size() >= batch_size:
            return True

        return False

    def retract_decode(self):
        sorted_ids = [i for i in range(len(self.requests))]
        sorted_ids.sort(
            key=lambda i: (
                len(self.requests[i].output_ids),
                -len(self.requests[i].input_ids),
            ),
            reverse=True,
        )

        retracted_reqs = []
        seq_lens_cpu = self.seq_lens.cpu().numpy()

        first_iter = True
        while (
            self.kv_pool.available_size() < len(sorted_ids) * self.retract_decode_steps
            or first_iter
        ):
            if len(sorted_ids) == 1:
                if self.kv_pool.available_size() <= 0:
                    raise RuntimeError("No space left for only one request")

                break

            first_iter = True
            id = sorted_ids.pop()
            req = self.requests[id]
            retracted_reqs.append(req)

            last_uncached_pos = len(req.prefix_ids)
            token_ids = self.request_to_token_pool.request_to_token[
                req.request_pool_id, last_uncached_pos : seq_lens_cpu[id]
            ]
            self.kv_pool.free(token_ids)
            self.request_to_token_pool.free(req.request_pool_id)

            self.tree_cache.decrease_lock_ref(req.last_node)

            residual_size = (
                len(sorted_ids) * self.retract_decode_steps
                - self.kv_pool.available_size()
            )
            residual_size = max(0, residual_size)
            self.tree_cache.evict(residual_size, self.kv_pool.free)

            req.reset_for_retract()

        self.filter(keep_ids=sorted_ids)

        total_decoded_tokens = sum(len(req.output_ids) for req in self.requests)
        total_max_new_tokens = sum(
            req.sampling_params.max_new_tokens for req in self.requests
        )

        new_estimate_ratio = (
            total_decoded_tokens + self.retract_decode_steps * len(self.requests)
        ) / total_max_new_tokens
        new_estimate_ratio = max(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio


class ForwardBatch:
    def __init__(
        self,
        input_ids,
        kv_pool_ids,
        seq_lens,
        extend_lens,
        positions,
        mode,
    ):
        self.input_ids = input_ids
        self.kv_pool_ids = kv_pool_ids
        self.seq_lens = seq_lens
        self.extend_lens = extend_lens
        self.positions = positions
        self.mode = mode

        self.seq_lens_max = seq_lens.max().item()
