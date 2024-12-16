from pydantic import StrictInt
from slickconf import Config, Instance, MainConfig

from halite.transformers.infer.engine.batch import (
    SamplingParams as EngineSamplingParams,
)
from halite.projects.common.config import Model


class Fewshot(Config):
    sampler: Instance
    samples: Instance


class SamplingParams(Config):
    max_new_tokens: StrictInt | None = None
    min_new_tokens: StrictInt | None = 0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: StrictInt = -1
    min_p: float = 0
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None

    def build(
        self, prefix_len: int | None = None, max_context_len: int | None = None
    ) -> EngineSamplingParams:
        max_new_tokens = self.max_new_tokens
        if self.max_new_tokens is None:
            max_new_tokens = max_context_len - prefix_len

        return EngineSamplingParams(
            max_new_tokens=max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
        )


class Task(Config):
    name: str
    dataset: Instance
    evaluate_fn: Instance
    sampling_params: SamplingParams
    preprocess: list[Instance] | None = None
    fewshot: Fewshot | None = None
    prefix: Instance | None = None


class Eval(Config):
    batch_size: StrictInt
    tokenizer: str | None = None


class EvalTaskConfig(MainConfig):
    tasks: list[Task]
    model: Model
    eval: Eval
