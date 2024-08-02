import time
from typing import List, Optional, Tuple, Union

import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter

from meshfn.distributed import ParallelMode
from meshfn.transformers.builder.model import get_model_dtype
from meshfn.transformers.builder.vllm.config import ModelConfig
from meshfn.transformers.builder.vllm.worker import Worker

_LOGGING_INTERVAL_SEC = 10


def create_engine_configs(engine_args, transformer_config):
    model_config = ModelConfig(
        transformer_config,
        engine_args.tokenizer,
        engine_args.tokenizer_mode,
        engine_args.trust_remote_code,
        engine_args.download_dir,
        engine_args.load_format,
        engine_args.dtype,
        engine_args.seed,
    )
    cache_config = CacheConfig(
        engine_args.block_size,
        engine_args.gpu_memory_utilization,
        engine_args.swap_space,
    )
    parallel_config = ParallelConfig(
        engine_args.pipeline_parallel_size,
        engine_args.tensor_parallel_size,
        engine_args.worker_use_ray,
    )
    scheduler_config = SchedulerConfig(
        engine_args.max_num_batched_tokens,
        engine_args.max_num_seqs,
        model_config.get_max_model_len(),
    )

    return model_config, cache_config, parallel_config, scheduler_config


class vLLM(LLMEngine):
    def __init__(self, model, tokenizer, parallel_context=None, logger=None, **kwargs):
        dtype = None

        try:
            dtype = model.dtype

        except AttributeError:
            pass

        if dtype is None:
            dtype = get_model_dtype(model)

        tp_size = 1

        if parallel_context is not None:
            tp_size = parallel_context.world_size(ParallelMode.TENSOR_1D)

        engine_args = EngineArgs(
            model="meshfn",
            tokenizer="meshfn",
            tensor_parallel_size=tp_size,
            dtype=dtype,
            **kwargs,
        )
        engine_configs = create_engine_configs(engine_args, model.config)

        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger

        (
            self.model_config,
            self.cache_config,
            self.parallel_config,
            self.scheduler_config,
        ) = engine_configs
        self.log_stats = not engine_args.disable_log_stats
        self.request_counter = Counter()
        self.seq_counter = Counter()

        self._init_workers()
        self._init_cache()
        self.scheduler = Scheduler(self.scheduler_config, self.cache_config)

        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []

    def eval(self):
        self.model.eval()

    def _init_workers(self):
        self.workers = []
        worker = Worker(
            self.model, self.model_config, self.parallel_config, self.scheduler_config
        )
        self.workers.append(worker)

    def generate(self, prompts=None, sampling_params=None, prompt_token_ids=None):
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be provided.")

        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if prompts is not None and prompt_token_ids is not None:
            if len(prompts) != len(prompt_token_ids):
                raise ValueError(
                    "The lengths of prompts and prompt_token_ids " "must be the same."
                )
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # Add requests to the engine.
        if prompts is not None:
            num_requests = len(prompts)
        else:
            num_requests = len(prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None

            if isinstance(sampling_params, SamplingParams):
                params = sampling_params

            else:
                params = sampling_params[i]

            if prompt_token_ids is None:
                token_ids = None
            else:
                token_ids = prompt_token_ids[i]

            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()

            self._add_request(prompt, params, token_ids)

        return self._run_engine()

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
    ) -> None:
        request_id = str(next(self.request_counter))
        self.add_request(request_id, prompt, sampling_params, prompt_token_ids)

    def _run_engine(self) -> List[RequestOutput]:
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.has_unfinished_requests():
            step_outputs = self.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # FIXME(woosuk): Change to debug log.
        if self.logger is not None:
            self.logger.info(
                f"# GPU blocks: {num_gpu_blocks}, " f"# CPU blocks: {num_cpu_blocks}"
            )

        if num_gpu_blocks <= 0:
            raise ValueError(
                "No available memory for the cache blocks. "
                "Try increasing `gpu_memory_utilization` when "
                "initializing the engine."
            )

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    def _log_system_stats(
        self,
        prompt_run: bool,
        num_batched_tokens: int,
    ) -> None:
        now = time.time()
        # Log the number of batched input tokens.
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))

        elapsed_time = now - self.last_logging_time
        if elapsed_time < _LOGGING_INTERVAL_SEC:
            return

        # Discard the old stats.
        self.num_prompt_tokens = [
            (t, n) for t, n in self.num_prompt_tokens if now - t < _LOGGING_INTERVAL_SEC
        ]
        self.num_generation_tokens = [
            (t, n)
            for t, n in self.num_generation_tokens
            if now - t < _LOGGING_INTERVAL_SEC
        ]

        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = self.scheduler.block_manager.get_num_free_gpu_blocks()
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = self.scheduler.block_manager.get_num_free_cpu_blocks()
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        if self.logger is not None:
            self.logger.info(
                "Avg prompt throughput: "
                f"{avg_prompt_throughput:.1f} tokens/s, "
                "Avg generation throughput: "
                f"{avg_generation_throughput:.1f} tokens/s, "
                f"Running: {len(self.scheduler.running)} reqs, "
                f"Swapped: {len(self.scheduler.swapped)} reqs, "
                f"Pending: {len(self.scheduler.waiting)} reqs, "
                f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%"
            )
        self.last_logging_time = now
