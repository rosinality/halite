from collections.abc import Sequence
from collections import defaultdict
from typing import Any, Callable

import torch

from halite.transformers.infer.engine.engine import InferenceEngine


class Handler:
    def __init__(
        self,
        name: str,
        fn: Callable[[Any], torch.Tensor],
        args: tuple[str] | dict[str, str] = ("output_texts",),
        targets: str | list[str] = "*",
        preprocess: Callable[[Any], Any] | None = None,
        reward_padding: int = 0,
    ):
        self.name = name
        self.fn = fn
        self.args = args
        self.targets = targets
        self.preprocess = preprocess
        self.reward_padding = reward_padding


class RewardRegistry:
    def __init__(
        self,
        *args: list[Handler],
        postprocess: Callable[..., torch.Tensor] | None = None,
    ):
        self.handlers = args
        self.postprocess = postprocess

        if len(self.handlers) > 1 and self.postprocess is None:
            raise ValueError(
                "postprocess must be provided if there are multiple handlers"
            )

    def __call__(self, data: list[dict[str, Any]], types: list[str]) -> torch.Tensor:
        rewards_dict = {}

        for handler in self.handlers:
            handler_inputs = defaultdict(list)
            handler_ids = []

            for i, (row, type) in enumerate(zip(data, types)):
                if type in handler.targets or handler.targets == "*":
                    for key in handler.args:
                        handler_inputs[key].append(row[key])

                    handler_ids.append(i)

            handler_inputs = [handler_inputs[key] for key in handler.args]
            handler_output = handler.fn(*handler_inputs)

            if handler_output.ndim == 1:
                handler_rewards = handler_output.new_full(
                    (len(data),), handler.reward_padding
                )

            else:
                handler_rewards = handler_output.new_full(
                    (len(data), handler_output.shape[1]), handler.reward_padding
                )

            handler_rewards[handler_ids] = handler_output

            rewards_dict[handler.name] = handler_rewards

        if self.postprocess is not None:
            rewards = self.postprocess(rewards_dict)

        else:
            rewards = next(iter(rewards_dict.values()))

        return rewards, rewards_dict


class Detokenize:
    def __init__(
        self,
        tokenizer: Any,
        keys=("output_ids",),
        output_keys=("output_texts",),
        **tokenizer_kwargs,
    ):
        self.tokenizer = tokenizer
        self.keys = keys
        self.output_keys = output_keys
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, data):
        for i, key in enumerate(self.keys):
            if key not in data:
                continue

            target_key = self.output_keys[i] if self.output_keys else key

            detokenized = [
                self.tokenizer.decode(row, **self.tokenizer_kwargs) for row in data[key]
            ]

            data[target_key] = detokenized

        return data


class RolloutGenerator:
    def __init__(
        self,
        inference_engine: InferenceEngine,
        reward_registry: RewardRegistry,
        preprocessors: list[Callable[[dict], dict]] | None = None,
    ):
        self.inference_engine = inference_engine
        self.reward_registry = reward_registry
        self.preprocessors = preprocessors if preprocessors is not None else []

    def initialize(self):
        self.inference_engine.initialize()

    def cleanup(self):
        self.inference_engine.cleanup()

    def load_state_dict(self, state_dict, assign=True):
        self.inference_engine.load_state_dict(state_dict, assign=assign)

    def generate(self, requests, types, batch):
        samples = self.inference_engine.infer_batch(requests)

        data = []
        for sample in samples:
            data.append(sample.to_dict())

        for i in range(len(data)):
            row = data[i]
            batch_row = batch.slice(i)

            assert set(batch_row.keys()).isdisjoint(
                set(row.keys())
            ), "duplicate key found between samples and batch"

            row.update(batch_row)

        preprocessed_data = []

        for sample in data:
            for preprocessor in self.preprocessors:
                sample = preprocessor(sample)

            preprocessed_data.append(sample)

        rewards, rewards_dict = self.reward_registry(preprocessed_data, types)

        return samples, rewards, rewards_dict
