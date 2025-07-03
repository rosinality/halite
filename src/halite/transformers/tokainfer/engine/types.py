import math
from dataclasses import dataclass, field
from typing import Any
from queue import Empty

from slickconf import instantiate
import torch.multiprocessing as mp

from halite.transformers.tokainfer.types import ServerConfig
from halite.transformers.tokainfer.engine.allocator import (
    BatchIndexAllocator,
    BlockAllocator,
)
from halite.transformers.tokainfer.engine.monitoring import StatsTracker
from halite.transformers.tokainfer.engine.stopping_predictor import EarlyStoppingTracker
from halite.transformers.tokainfer.types import RequestOutput
from halite.transformers.tokainfer.server_types import (
    SamplingParams,
    TokasaurusRequest,
)


@dataclass
class Sequence:
    id: str
    completion_total: int
    input_ids: list[int]

    sampling_params: SamplingParams = field(
        default_factory=lambda: SamplingParams(temperature=0.0, top_p=1.0)
    )
    stop: list[str] = field(default_factory=list)

    # tracks how many prefill/decoding tokens have
    # been scheduled (maybe not actually ran by the model yet)
    # note that the finishing the last prompt token also generates
    # the first completion token.
    prompt_scheduled: int = 0
    completion_scheduled: int = 0

    batch_index: int | None = None
    kv_indices: list[int] | None = None

    # the tokens/logprobs that have actually come back from the model
    # note that these lists will lag completion_scheduled because we
    # are asynchronously sending work to the model
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)

    num_cached_prompt_tokens: int | None = None

    cancelled: bool = False

    _expected_completion_length: int | None = None
    _expected_completion_length_with_buffer: int | None = None

    request: TokasaurusRequest | None = None
    output: RequestOutput | None = None

    def __repr__(self):
        return f"Seq(idx={self.id}, pre={self.prompt_scheduled}/{self.prompt_total()}, dec={self.completion_scheduled}/{self.completion_total})"

    def __hash__(self):
        return hash(self.id)

    def prompt_total(self):
        return len(self.input_ids)

    def total_scheduled(self):
        return self.prompt_scheduled + self.completion_scheduled

    def prompt_to_schedule(self):
        return self.prompt_total() - self.prompt_scheduled

    def expected_completion_length(self, add_buffer: bool = False):
        if add_buffer:
            val = self._expected_completion_length_with_buffer
        else:
            val = self._expected_completion_length

        if val is None:
            return self.completion_total

        assert val <= self.completion_total, (val, self.completion_total)
        return val

    def expected_completion_to_schedule(self, add_buffer: bool = False):
        return self.expected_completion_length(add_buffer) - self.completion_scheduled

    def expected_total_length(self, add_buffer: bool = False):
        return self.prompt_total() + self.expected_completion_length(add_buffer)

    def expected_num_additional_blocks(self, page_size: int, add_buffer: bool = False):
        # -1 since the last generated token isn't sent through the model, so we
        # don't need to reserve space for it
        kv_tokens_needed = (
            self.prompt_total() + self.expected_completion_length(add_buffer) - 1
        )

        if self.kv_indices is not None:
            kv_tokens_needed -= page_size * len(self.kv_indices)

        return math.ceil(kv_tokens_needed / page_size)

    def expected_last_page_len(self, page_size: int, add_buffer: bool = False):
        kv_tokens_needed = (
            self.prompt_total() + self.expected_completion_length(add_buffer) - 1
        )

        last_page_len = kv_tokens_needed % page_size
        if last_page_len == 0:
            last_page_len = page_size

        return last_page_len

    def max_num_additional_blocks(self, page_size: int):
        kv_tokens_needed = self.prompt_total() + self.completion_total - 1
        if self.kv_indices is not None:
            kv_tokens_needed -= page_size * len(self.kv_indices)

        return math.ceil(kv_tokens_needed / page_size)

    def most_recent_completion_ids(self, num_to_return: int):
        recently_decoded = self.completion_ids[-num_to_return:]
        return recently_decoded

    def num_uncached_prompt_tokens(self):
        assert self.num_cached_prompt_tokens is not None
        return self.prompt_total() - self.num_cached_prompt_tokens

    def num_cached_blocks(self, page_size: int):
        assert self.num_cached_prompt_tokens is not None
        assert self.num_cached_prompt_tokens % page_size == 0
        return self.num_cached_prompt_tokens // page_size

    def cached_blocks(self, page_size: int):
        assert self.kv_indices is not None
        num_cached_blocks = self.num_cached_blocks(page_size)
        return self.kv_indices[:num_cached_blocks]

    def uncached_blocks(self, page_size: int):
        assert self.kv_indices is not None
        num_cached_blocks = self.num_cached_blocks(page_size)
        return self.kv_indices[num_cached_blocks:]


@dataclass
class SchedulingQueue:
    """
    NOTE: this keeps track of the state ACCORDING TO THE SCHEDULER - may not
    actually reflect the true state of requests. E.g., a "finished" sequence
    here may still have outstanding tokens to actually compute, and may not
    yet have had its request returned to the user.


    Also, we're using the fact that Python dicts are ordered, so that these dicts
    are insertion-order queues that we can also index by id.
    """

    decoding_seqs: dict[str, Sequence] = field(default_factory=dict)
    prefilling_seqs: dict[str, Sequence] = field(default_factory=dict)
    queued_seqs: dict[str, Sequence] = field(default_factory=dict)

    def initialize(self):
        self.decoding_seqs = {}
        self.prefilling_seqs = {}
        self.queued_seqs = {}

    def cleanup(self):
        self.initialize()

    def get(self, sid: str):
        out = (
            self.decoding_seqs.get(sid)
            or self.prefilling_seqs.get(sid)
            or self.queued_seqs.get(sid)
        )
        if out is None:
            raise ValueError(f"Request {sid} not found")
        return out

    def __getitem__(self, sid: str):
        return self.get(sid)

    def get_decoding(self, sid: str):
        return self.decoding_seqs[sid]

    def get_prefilling(self, sid: str):
        return self.prefilling_seqs[sid]

    def get_queued(self, sid: str):
        return self.queued_seqs[sid]

    def add_decoding(self, seq: Sequence):
        self.decoding_seqs[seq.id] = seq

    def add_prefilling(self, seq: Sequence):
        self.prefilling_seqs[seq.id] = seq

    def add_queued(self, seq: Sequence):
        self.queued_seqs[seq.id] = seq

    def remove_decoding(self, sid: str):
        self.decoding_seqs.pop(sid)

    def remove_prefilling(self, sid: str):
        self.prefilling_seqs.pop(sid)

    def remove_queued(self, sid: str):
        self.queued_seqs.pop(sid)

    def remove(self, sid: str):
        if self.in_decoding(sid):
            self.remove_decoding(sid)
        elif self.in_prefilling(sid):
            self.remove_prefilling(sid)
        elif self.in_queued(sid):
            self.remove_queued(sid)
        else:
            raise ValueError(f"Sequence {sid} not found")

    def in_decoding(self, sid: str) -> bool:
        return sid in self.decoding_seqs

    def in_prefilling(self, sid: str) -> bool:
        return sid in self.prefilling_seqs

    def in_queued(self, sid: str) -> bool:
        return sid in self.queued_seqs

    def insert_at_head_of_queued(self, seqs: list[Sequence]):
        for d in [self.decoding_seqs, self.prefilling_seqs, self.queued_seqs]:
            for seq in seqs:
                assert seq.id not in d

        new_queued = {}
        for seq in seqs:
            new_queued[seq.id] = seq

        new_queued.update(self.queued_seqs)
        self.queued_seqs = new_queued

    def running_seqs(self):
        return list(self.decoding_seqs.values()) + list(self.prefilling_seqs.values())

    def unfinished_seqs(self):
        return self.running_seqs() + list(self.queued_seqs.values())

    def num_unfinished_seqs(self):
        return (
            len(self.decoding_seqs) + len(self.prefilling_seqs) + len(self.queued_seqs)
        )

    def num_running_seqs(self):
        return len(self.decoding_seqs) + len(self.prefilling_seqs)


@dataclass
class ScheduleDecision:
    id: str
    decoding_seqs: list[Sequence]
    prefill_seqs: list[tuple[Sequence, int]]  # (seq, prefill_length)

    def __post_init__(self):
        # construct the list of ids where we expect to get tokens
        # this order is important, it's what the model returns to us:
        # - first any prefill seqs (in order of prefill_seqs)
        # - then any decode seqs
        self.seqs_with_tokens_to_return = list[Sequence]()

        for seq, prefill_len in self.prefill_seqs:
            assert (
                seq.prompt_to_schedule() >= prefill_len
            ), f"Request {seq.id} has {seq.prompt_to_schedule()} prompt tokens to schedule but prefill length is {prefill_len}"

            # if we finish prefilling, the model will sample for us
            if seq.prompt_to_schedule() == prefill_len:
                self.seqs_with_tokens_to_return.append(seq)

        self.seqs_with_tokens_to_return.extend(self.decoding_seqs)

    def num_decoding_tokens(self):
        return len(self.decoding_seqs)

    def num_prefill_tokens(self):
        return sum(prefill_len for _, prefill_len in self.prefill_seqs)

    def batch_size(self):
        return self.num_decoding_tokens() + self.num_prefill_tokens()

    def num_seqs(self):
        return len(self.decoding_seqs) + len(self.prefill_seqs)


def clear_queue(queue):
    try:
        while True:
            queue.get_nowait()

    except Empty:
        pass


import os


@dataclass
class ManagerState:
    tokenizer: Any
    config: ServerConfig
    block_allocator: "BlockAllocator"
    batch_index_allocator: "BatchIndexAllocator"
    q_manager_to_model: mp.Queue
    q_model_to_manager: mp.Queue
    q_server_to_manager: mp.Queue
    q_manager_to_server: mp.Queue
    process_name: str
    scheduling_queue: SchedulingQueue = field(default_factory=SchedulingQueue)
    inflight_schedule_decisions: dict[str, ScheduleDecision] = field(
        default_factory=dict
    )
    _inflight_schedule_decisions: dict[str, ScheduleDecision] = field(
        default_factory=dict
    )
    finished_seq_ids: set[str] = field(default_factory=set)
    bumped_seq_id_to_ttl: dict[str, int] = field(default_factory=dict)
    num_inflight_batches: int = 0
    stats_tracker: StatsTracker = field(default_factory=StatsTracker)
    early_stopping_tracker: EarlyStoppingTracker | None = None
    req_id_to_seq_ids: dict[str, set[str]] = field(default_factory=dict)
    req_ids_to_cancel: set[str] = field(default_factory=set)
    sent_no_more_inputs: bool = False
    last_step_num_prefill: int = 1024 * 1024 * 1024

    def __post_init__(self):
        self.tokenizer = instantiate(self.tokenizer)

    def deallocate(self, seq: Sequence):
        assert seq.kv_indices is not None

        # NOTE: the last decoded token is NOT sent back through the model, so we don't
        # generate a KV cache for it (and thus can't prefix cache with it).
        ids_for_update = seq.input_ids + seq.completion_ids[:-1]
        if seq.prompt_scheduled < seq.prompt_total():
            ids_for_update = ids_for_update[: seq.prompt_scheduled]

        self.block_allocator.free_and_update(seq.id, seq.kv_indices, ids_for_update)
        seq.kv_indices = None

        assert seq.batch_index is not None
        self.batch_index_allocator.free(seq.batch_index)
        seq.batch_index = None

    def initialize(self):
        self.scheduling_queue.initialize()
        self.block_allocator.initialize()
        self.batch_index_allocator.initialize()

        if self.early_stopping_tracker is not None:
            self.early_stopping_tracker.initialize()

        self._reset_fields()
        self._inflight_schedule_decisions = {}

    def _reset_fields(self):
        self.inflight_schedule_decisions = {}
        self.finished_seq_ids = set()
        self.bumped_seq_id_to_ttl = {}
        self.num_inflight_batches = 0
        # self.stats_tracker.reset()
        self.req_id_to_seq_ids = {}
        self.req_ids_to_cancel = set()
        self.sent_no_more_inputs = False

    def cleanup(self):
        self.scheduling_queue.cleanup()
        self.block_allocator.cleanup()
        self.batch_index_allocator.cleanup()

        if self.early_stopping_tracker is not None:
            self.early_stopping_tracker.cleanup()

        self._reset_fields()


@dataclass
class HydragenGroup:
    block_ids: list[int]
    seq_ids: set[str]
