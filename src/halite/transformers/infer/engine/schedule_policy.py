from contextlib import contextmanager
from enum import Enum, auto


class AddRequestResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No token left
    OTHER = auto()  # Other reasons to stop adding requests


class PrefillAdder:
    def __init__(
        self,
        tree_cache,
        running_batch,
        new_token_ratio,
        remaining_total_tokens,
        remaining_input_tokens,
        remaining_chunk_tokens,
        mixed_with_decode_tokens: int = 0,
    ):
        self.tree_cache = tree_cache
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.remaining_total_tokens = remaining_total_tokens - mixed_with_decode_tokens
        self.remaining_input_tokens = remaining_input_tokens - mixed_with_decode_tokens
        self.remaining_chunk_tokens = remaining_chunk_tokens

        if self.remaining_chunk_tokens is not None:
            self.remaining_chunk_tokens -= mixed_with_decode_tokens

        self.current_remaining_tokens = (
            remaining_total_tokens - mixed_with_decode_tokens
        )

        self.request_states = None
        self.can_run_list = []
        self.new_inflight_requst = None

        if running_batch is not None:
            self.remaining_total_tokens -= sum(
                [
                    min(req.sampling_params.max_new_tokens - len(req.output_ids), 4096)
                    * self.new_token_ratio
                    for req in running_batch.requests
                ]
            )

    def budget_state(self):
        if self.remaining_total_tokens <= 0 or self.current_remaining_tokens <= 0:
            return AddRequestResult.NO_TOKEN

        if self.remaining_input_tokens <= 0 or (
            self.remaining_chunk_tokens is not None and self.remaining_chunk_tokens <= 0
        ):
            return AddRequestResult.OTHER

        return AddRequestResult.CONTINUE

    def prefill_request(self, prefix_len, extend_input_len, max_new_tokens):
        self.remaining_total_tokens -= extend_input_len + max_new_tokens
        self.current_remaining_tokens -= extend_input_len
        self.remaining_input_tokens -= extend_input_len

        if self.remaining_chunk_tokens is not None:
            self.remaining_chunk_tokens -= extend_input_len

    @contextmanager
    def lock_node(self, last_node):
        try:
            delta = self.tree_cache.increase_lock_ref(last_node)
            self.remaining_total_tokens += delta

            yield None

        finally:
            delta = self.tree_cache.decrease_lock_ref(last_node)
            self.remaining_total_tokens += delta

    def add_request(self, request):
        total_tokens = request.extend_input_len + min(
            request.sampling_params.max_new_tokens, 4096
        )
        input_tokens = request.extend_input_len
        prefix_len = len(request.prefix_ids)

        if total_tokens > self.remaining_total_tokens:
            return AddRequestResult.NO_TOKEN

        if input_tokens > self.remaining_input_tokens and len(self.can_run_list) != 0:
            return AddRequestResult.OTHER

        with self.lock_node(request.last_node):
            if total_tokens > self.remaining_total_tokens:
                return AddRequestResult.NO_TOKEN

            if (
                self.remaining_chunk_tokens is None
                or input_tokens <= self.remaining_chunk_tokens
                or (
                    request.return_logprob and request.normalized_prompt_logprob is None
                )
            ):
                self.can_run_list.append(request)
                self.tree_cache.increase_lock_ref(request.last_node)
                self.prefill_request(
                    prefix_len,
                    input_tokens,
                    min(request.sampling_params.max_new_tokens, 4096),
                )

        return self.budget_state()
