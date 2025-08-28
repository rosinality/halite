from collections import defaultdict
from dataclasses import dataclass, field, fields, replace
from random import Random
from tarfile import LinkOutsideDestinationError
from typing import Any, Callable, NamedTuple
import uuid

import torch

from halite.data.record import Record
from halite.projects.common.param import Param
from halite.transformers.infer.engine.engine import InferenceEngine
from halite.transformers.infer.types import InferenceRequest


class Response(NamedTuple):
    id: str
    responses: list[list[int]]
    response_logprobs: list[list[float]] | None = None


@dataclass
class Rollout:
    id: str
    input_ids: list[int]
    output_ids: list[int] | None = None
    output_logprobs: list[float] | None = None
    type: str | None = None
    rewards: torch.Tensor | list[float] | None = None
    rewards_dict: dict[str, torch.Tensor | float] | None = None
    sampling_params: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def to_inference_request(self) -> InferenceRequest:
        return InferenceRequest(
            id=self.id,
            input_ids=self.input_ids,
            sampling_params=self.sampling_params,
        )

    def _fields(self):
        return [field.name for field in fields(self)]

    def has_field(self, key: str):
        return key in self._fields() or key in self.state

    def get_field(self, key: str):
        if key in self._fields():
            return getattr(self, key)

        else:
            return self.state[key]

    def set_field(self, key: str, value: Any):
        if key in self._fields():
            raise AttributeError(f"name `{key}` is overlaps with reserved fields")

        else:
            self.state[key] = value

    def copy(self):
        return Rollout(
            id=self.id,
            input_ids=self.input_ids,
            output_ids=self.output_ids,
            output_logprobs=self.output_logprobs,
            type=self.type,
            rewards=self.rewards,
            rewards_dict=self.rewards_dict,
            sampling_params=self.sampling_params,
            state=self.state.copy(),
        )


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

    def __call__(self, rollouts: list[Rollout]) -> torch.Tensor:
        rewards_dict = {}

        for handler in self.handlers:
            handler_inputs = defaultdict(list)
            handler_ids = []

            for i, rollout in enumerate(rollouts):
                type = rollout.type

                if type in handler.targets or handler.targets == "*":
                    for key in handler.args:
                        handler_inputs[key].append(rollout.get_field(key))

                    handler_ids.append(i)

            handler_inputs = [handler_inputs[key] for key in handler.args]
            handler_output = handler.fn(*handler_inputs)

            if handler_output.ndim == 1:
                handler_rewards = handler_output.new_full(
                    (len(rollouts),), handler.reward_padding
                )

            else:
                handler_rewards = handler_output.new_full(
                    (len(rollouts), handler_output.shape[1]), handler.reward_padding
                )

            handler_rewards[handler_ids] = handler_output

            rewards_dict[handler.name] = handler_rewards

        if self.postprocess is not None:
            rewards, rewards_dict = self.postprocess(rewards_dict, rollouts)

        else:
            rewards = next(iter(rewards_dict.values()))

        rewards_dict_per_rollout = {k: v.unbind(0) for k, v in rewards_dict.items()}
        rewards_per_rollout = rewards.unbind(0)

        for i, (rollout, reward) in enumerate(zip(rollouts, rewards_per_rollout)):
            rollout.rewards = reward
            rollout.rewards_dict = {
                k: v[i] for k, v in rewards_dict_per_rollout.items()
            }

        return rollouts


class RequestBuilder:
    def __init__(
        self,
        prompt_key: str,
        sampling_params: dict[str, Any],
        tokenizer: Any | None = None,
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

        self.tokenizer = tokenizer
        self.type = type
        self.type_key = type_key
        self.meta_maps = meta_maps
        self.random = Random(seed)

    def __call__(self, batch: Record):
        target_keys = {self.prompt_key, *self.meta_maps.values()}
        if self.type_key is not None:
            target_keys.add(self.type_key)

        rollouts = []
        records = batch.unbind(target_keys)

        if self.tokenizer is not None:
            prompts = self.tokenizer.encode_batch(
                [record[self.prompt_key] for record in records]
            )

        else:
            prompts = [record[self.prompt_key] for record in records]

        for prompt, record in zip(prompts, records):
            sampling_param = self.sample_sampling_params()

            if self.type is not None:
                type = self.type

            else:
                type = record[self.type_key]

            state = {k: record[v] for k, v in self.meta_maps.items()}

            rollouts.append(
                Rollout(
                    id=uuid.uuid4().hex,
                    input_ids=prompt,
                    type=type,
                    sampling_params=sampling_param,
                    state=state,
                )
            )

        return rollouts

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
        preprocessors: list[Callable[[Rollout], Rollout]] | None = None,
        finish_condition: Callable[[Rollout], bool] | None = None,
    ):
        self.inference_engine = inference_engine
        self.reward_registry = reward_registry
        self.preprocessors = preprocessors if preprocessors is not None else []
        self.finish_condition = finish_condition

    def initialize(self):
        self.inference_engine.initialize()

    def cleanup(self):
        self.inference_engine.cleanup()

    def load_state_dict(self, state_dict, assign=True):
        self.inference_engine.load_state_dict(state_dict, assign=assign)

    def generate(self, rollouts: list[Rollout]) -> list[Rollout]:
        infer_requests = [rollout.to_inference_request() for rollout in rollouts]

        samples = self.inference_engine.infer_batch(infer_requests)

        samples_map = {sample.id: sample for sample in samples}
        output_ids = []

        for rollout in rollouts:
            output_ids.append(samples_map[rollout.id].output_ids)

        return self.build_rollout(rollouts, output_ids)

    def build_rollout(
        self,
        rollouts: list[Rollout],
        output_ids: list[list[list[int]]],
        output_logprobs: list[list[list[float]]] | None = None,
    ) -> list[Rollout]:
        rollouts_expand = []

        for i, (rollout, output_id) in enumerate(zip(rollouts, output_ids)):
            for output_i, output in enumerate(output_id):
                rollout_copy = rollout.copy()
                rollout_copy.output_ids = output

                if output_logprobs is not None:
                    rollout_copy.output_logprobs = output_logprobs[i][output_i]

                rollouts_expand.append(rollout_copy)

        preprocessed_data = []

        for rollout in rollouts_expand:
            for preprocessor in self.preprocessors:
                rollout = preprocessor(rollout)

            preprocessed_data.append(rollout)

        rollouts = self.reward_registry(preprocessed_data)

        return rollouts
