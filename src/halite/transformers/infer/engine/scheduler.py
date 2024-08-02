from dataclasses import dataclass
from typing import Sequence

import torch

from halite.logging import logger

from halite.transformers.infer.batch import Request, SamplingParams, Batch
from halite.transformers.infer.model_runner import ModelRunner
from halite.transformers.infer.schedule_policy import PrefillAdder, AddRequestResult
from halite.transformers.infer.radix_cache import RadixCache


class ServerArgs:
    max_prefill_tokens: int = 16384
    chunked_prefill_size: int = 8192

    default_init_new_token_ratio: float = 0.7
    default_min_new_token_ratio_factor: float = 0.14
    default_new_token_ratio_decay_steps: int = 600
    schedule_conservativeness: float = 1.0


class Scheduler:
    def __init__(self, model_runner: ModelRunner, tokenizer, server_args: ServerArgs):
        self.model_runner = model_runner
        self.tokenizer = tokenizer

        self.request_to_token_pool = self.model_runner.request_to_token_pool
        self.kv_pool = self.model_runner.kv_pool

        self.tree_cache = RadixCache(self.request_to_token_pool, self.kv_pool)

        self.waiting_queue = []
        self.running_batch = None
        self.current_batch = None

        self.max_running_requests = self.model_runner.max_running_requests
        self.is_mixed_chunk = False

        self.init_new_token_ratio = min(
            server_args.default_init_new_token_ratio
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * server_args.default_min_new_token_ratio_factor,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / server_args.default_new_token_ratio_decay_steps
        self.new_token_ratio = self.init_new_token_ratio
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.chunked_prefill_size = server_args.chunked_prefill_size

        self.last_batch = None

        self.n_generated_tokens = 0
        self.n_decode_steps_per_batch = 1

    def infer_batch(self, requests):
        outputs = []

        finished, batch_output = self.infer(requests)

        while not finished:
            finished, batch_output = self.infer([])
            outputs.extend(batch_output)

        outputs_sorted = sorted(outputs, key=lambda x: x[0])

        return [x[1:] for x in outputs_sorted]

    def infer(self, requests):
        self.process_requests(range(len(requests)), requests)
        batch = self.get_next_batch()
        self.current_batch = batch
        finished = batch is None
        batch_output = []

        if batch:
            result = self.run_batch(batch)
            batch_output = self.process_batch_result(batch, result)

            if batch.mode.is_decode():
                for _ in range(self.n_decode_steps_per_batch - 1):
                    if not self.running_batch:
                        break

                    self.update_running_batch()

                    if not self.running_batch:
                        break

                    result = self.run_batch(batch)
                    self.process_batch_result(batch, result)

        else:
            self.check_memory()
            self.new_token_ratio = self.init_new_token_ratio

        self.last_batch = batch

        return finished, batch_output

    def check_memory(self):
        available_size = self.kv_pool.available_size() + self.tree_cache.evictable_size

        if available_size != self.model_runner.max_total_tokens:
            logger.warning(
                f"Available memory is {available_size}, but max total tokens is {self.model_runner.max_total_tokens}"
            )

        if (
            len(self.request_to_token_pool.free_slots)
            != self.request_to_token_pool.max_size
        ):
            logger.warning(
                f"Available request slots is {len(self.request_to_token_pool.free_slots)}, but request pool size is {self.request_to_token_pool.max_size}"
            )

    def build_batch_request(self, requests):
        results = []

        for i, req in enumerate(requests):
            if isinstance(req, str):
                req = Request(i, req, self.tokenizer.encode(req), SamplingParams())

            elif isinstance(req, Sequence):
                text_or_tokens, sampling_params = req

                if isinstance(text_or_tokens, str):
                    input_text = text_or_tokens
                    input_ids = self.tokenizer.encode(text_or_tokens)

                else:
                    input_text = None
                    input_ids = text_or_tokens

                req = Request(i, input_text, input_ids, sampling_params)

            else:
                raise ValueError(f"Invalid request type: {type(req)}")

            results.append(req)

        return results

    def process_requests(self, ids, requests):
        for id, req in zip(ids, requests):
            self.handle_generate_request(id, req)

    def handle_generate_request(self, id, req):
        req = Request(id, req.input_text, req.input_ids, req.sampling_params)

        self.waiting_queue.append(req)

    def get_next_batch(self):
        if (
            self.last_batch
            and not self.last_batch.mode.is_decode()
            and not self.last_batch.is_empty()
        ):
            if not self.last_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = self.last_batch

                else:
                    self.running_batch.merge(self.last_batch)

        new_batch = self.get_new_prefill_batch()
        if new_batch is not None:
            return new_batch

        if self.running_batch is None:
            return

        before_batch_size = self.running_batch.batch_size()
        self.update_running_batch()
        if not self.running_batch:
            self.batch_is_full = False

            return None

        if before_batch_size != self.running_batch.batch_size():
            self.batch_is_full = False

        return self.running_batch

    def get_new_prefill_batch(self):
        running_batch_size = (
            len(self.running_batch.requests) if self.running_batch else 0
        )
        if running_batch_size >= self.max_running_requests:
            self.batch_is_full = True

            return None

        adder = PrefillAdder(
            self.tree_cache,
            self.running_batch,
            self.new_token_ratio,
            self.kv_pool.available_size() + self.tree_cache.evictable_size,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_batch_size if self.is_mixed_chunk else 0,
        )

        for request in self.waiting_queue:
            if (
                running_batch_size + len(adder.can_run_list)
                >= self.max_running_requests
            ):
                self.batch_is_full = True

                break

            request.prepare_next_input(self.tree_cache)
            result = adder.add_request(request)

            if result != AddRequestResult.CONTINUE:
                if result == AddRequestResult.NO_TOKEN:
                    self.batch_is_full = True

                break

        can_run_list = adder.can_run_list
        if len(can_run_list) == 0:
            return None

        self.waiting_queue = [
            req for req in self.waiting_queue if req not in set(can_run_list)
        ]

        new_batch = Batch(
            can_run_list,
            request_to_token_pool=self.request_to_token_pool,
            kv_pool=self.kv_pool,
            tree_cache=self.tree_cache,
        )
        new_batch.prepare_for_extend()

        return new_batch

    def update_running_batch(self):
        batch = self.running_batch

        batch.filter()
        if batch.is_empty():
            self.running_batch = None

            return

        if not batch.check_decode_memory():
            # old_ratio = self.new_token_ratio
            retracted_reqs, new_token_ratio = batch.retract_decode()
            self.new_token_ratio = new_token_ratio

            self.waiting_queue.extend(retracted_reqs)

        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        batch.prepare_for_decode()

    def run_batch(self, batch):
        if batch.mode.is_decode() or batch.extend_n_tokens != 0:
            logits_output, next_token_ids = self.model_runner.forward_and_sample(batch)

        else:
            logits_output = None
            next_token_ids = torch.full((batch.batch_size(),), 0)

        batch.output_ids = next_token_ids

        return logits_output, next_token_ids, batch.id

    def process_batch_result(self, batch, result):
        if batch.mode.is_decode():
            out = self.process_batch_result_decode(batch, result)

            if batch.is_empty():
                self.running_batch = None

        else:
            out = self.process_batch_result_prefill(batch, result)

        return out

    def process_batch_result_decode(self, batch, result):
        logits_output, next_token_ids, batch_id = result
        self.n_generated_tokens += len(batch.requests)

        if batch.return_logprob:
            next_token_logprobs = logits_output.next_token_logprobs[
                torch.arange(len(next_token_ids), device=self.device),
                next_token_ids,
            ].tolist()

        next_token_ids = next_token_ids.tolist()

        self.kv_pool.free_group_begin()

        for i, (req, next_token_id) in enumerate(zip(batch.requests, next_token_ids)):
            req.completion_tokens_without_jump_forward += 1
            req.output_ids.append(next_token_id)
            req.check_finished(self.tokenizer)

            if req.finished():
                self.tree_cache.cache_finished_request(req)

            if req.return_logprob:
                req.output_logprobs.append((next_token_logprobs[i], next_token_id))
                if req.top_logprobs_sum > 0:
                    req.output_top_logprobs.append(logits_output.output_top_logprobs[i])

        output = self.get_output(batch.requests)

        self.kv_pool.free_group_end()

        return output

    def process_batch_result_prefill(self, batch, result):
        logits_output, next_token_ids, batch_id = result

        if batch.return_logprob:
            logits_output.next_token_logprobs = logits_output.next_token_logprobs[
                torch.arange(len(next_token_ids), device=self.device),
                next_token_ids,
            ].tolist()
            logits_output.input_token_logprobs = (
                logits_output.input_token_logprobs.tolist()
            )
            logits_output.normalized_prompt_logprob = (
                logits_output.normalized_prompt_logprob.item()
            )

        next_token_ids = next_token_ids.tolist()

        logprob_pt = 0
        for i, req in enumerate(batch.requests):
            req.completion_tokens_without_jump_forward += 1
            req.output_ids.append(next_token_ids[i])
            req.check_finished(self.tokenizer)

            if req.finished():
                self.tree_cache.cache_finished_request(req)

            elif not batch.decoding_requests or req not in batch.decoding_requests:
                self.tree_cache.cache_unfinished_request(req)

            if req.return_logprob:
                logprob_pt += self.add_logprob_return_values(
                    i, req, logprob_pt, next_token_ids, logits_output
                )

        return self.get_output(batch.requests)

    def get_output(self, requests):
        results = []

        for req in requests:
            if req.finished():
                results.append((req.id, req.input_ids, req.output_ids))

        return results
