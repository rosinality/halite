from collections import defaultdict
from random import Random
from typing import Any, Callable, NamedTuple

import torch

from halite.data.record import Record
from halite.projects.common.param import Param
from halite.transformers.infer.engine.engine import InferenceEngine


class Rollouts(NamedTuple):
    samples: list[dict[str, Any]]
    rewards: torch.Tensor
    rewards_dict: dict[str, torch.Tensor]
    sampling_params: list[dict[str, Any]]


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


class Request(NamedTuple):
    prompt: str
    type: str
    sampling_params: dict[str, Any] = {}
    meta: dict[str, Any] = {}


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
            rewards, rewards_dict = self.postprocess(rewards_dict, data, types)

        else:
            rewards = next(iter(rewards_dict.values()))

        return rewards, rewards_dict


class RequestBuilder:
    def __init__(
        self,
        prompt_key: str,
        sampling_params: dict[str, Any],
        type: str | None = None,
        type_key: str | None = None,
        meta_maps: dict[str, str] | None = None,
        seed=None,
    ):
        self.prompt_key = prompt_key
        self.sampling_params = sampling_params

        if type is not None and type_key is not None:
            raise ValueError("type and type_key cannot both be provided")

        if type is None and type_key is None:
            raise ValueError("type or type_key must be provided")

        self.type = type
        self.type_key = type_key
        self.meta_maps = meta_maps
        self.random = Random(seed)

    def __call__(self, batch: Record):
        target_keys = {self.prompt_key, *self.meta_maps.values()}
        if self.type_key is not None:
            target_keys.add(self.type_key)

        requests = []
        records = batch.unbind(target_keys)

        for record in records:
            prompt = record[self.prompt_key]
            sampling_param = self.sample_sampling_params()

            if self.type is not None:
                type = self.type

            else:
                type = record[self.type_key]

            meta = {k: record[v] for k, v in self.meta_maps.items()}

            requests.append(Request(prompt, type, sampling_param, meta))

        return requests

    def sample_sampling_params(self):
        sampled_params = {}

        for k, v in self.sampling_params.items():
            if isinstance(v, Param):
                sampled_params[k] = v.sample(self.random)

            else:
                sampled_params[k] = v

        return sampled_params


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

    def generate(self, requests: list[Request]):
        infer_requests = [
            [request.prompt, request.sampling_params] for request in requests
        ]

        samples = self.inference_engine.infer_batch(infer_requests)

        data = []
        for sample in samples:
            data.append(sample.to_dict())

        samples_expand = []
        request_types = []
        sampling_params = []

        for i in range(len(data)):
            row = data[i]
            batch_row = requests[i].meta

            assert set(batch_row.keys()).isdisjoint(
                set(row.keys())
            ), "duplicate key found between samples and batch"

            row.update(batch_row)

            for output in row["output_ids"]:
                expanded_row = row.copy()
                expanded_row["output_ids"] = output
                sampling_params.append(requests[i].sampling_params)
                samples_expand.append(expanded_row)
                request_types.append(requests[i].type)

        preprocessed_data = []

        for sample in samples_expand:
            for preprocessor in self.preprocessors:
                sample = preprocessor(sample)

            preprocessed_data.append(sample)

        rewards, rewards_dict = self.reward_registry(preprocessed_data, request_types)

        return Rollouts(preprocessed_data, rewards, rewards_dict, sampling_params)
