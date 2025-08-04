from collections.abc import Sequence
from dataclasses import fields
from typing import Any, Callable, NamedTuple, Protocol

import torch

from halite.projects.common.rollout import Rollout


class Batch(NamedTuple):
    input_ids: torch.Tensor
    response_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    rewards: torch.Tensor
    ids: torch.Tensor
    prompt_len: torch.Tensor
    response_len: torch.Tensor
    temperatures: torch.Tensor | float

    def split(self, batch_size):
        if self.input_ids.shape[0] % batch_size != 0:
            raise ValueError(
                f"batch size {batch_size} does not divide input_ids.shape[0] {self.input_ids.shape[0]}"
            )

        n_chunk = max(1, self.input_ids.shape[0] // batch_size)
        batches_dict = {}
        for k, v in self._asdict().items():
            if isinstance(v, torch.Tensor):
                batches_dict[k] = v.chunk(n_chunk, dim=0)

            else:
                batches_dict[k] = [v] * n_chunk

        batches = []
        for i in range(n_chunk):
            batches.append(Batch(**{k: v[i] for k, v in batches_dict.items()}))

        return batches


class RolloutBatch(NamedTuple):
    rollouts: list[Rollout]
    rewards: torch.Tensor
    batch: Batch | None = None
    returns: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    actor_log_probs: torch.Tensor | None = None
    actor_entropy: torch.Tensor | None = None

    def split(self, batch_size):
        if self.rewards.shape[0] % batch_size != 0:
            raise ValueError(
                f"batch size {batch_size} does not divide rewards.shape[0] {self.rewards.shape[0]}"
            )

        n_chunk = self.rewards.shape[0] // batch_size

        batches_dict = {}
        for k, v in self._asdict().items():
            if isinstance(v, torch.Tensor):
                batches_dict[k] = v.chunk(n_chunk, dim=0)

            elif hasattr(v, "split"):
                batches_dict[k] = v.split(batch_size)

            elif isinstance(v, Sequence):
                batches_dict[k] = [
                    v[i : i + batch_size] for i in range(0, len(v), batch_size)
                ]

            else:
                batches_dict[k] = [v] * n_chunk

        batches = []
        for i in range(n_chunk):
            batches.append(RolloutBatch(**{k: v[i] for k, v in batches_dict.items()}))

        return batches


class CriticMetrics(NamedTuple):
    rewards_mean: torch.Tensor
    advantage_mean: torch.Tensor
    advantage_var: torch.Tensor
    return_mean: torch.Tensor
    return_var: torch.Tensor


class ActorLossResults(Protocol):
    loss: torch.Tensor

    def metric_dict(self) -> dict[str, torch.Tensor]: ...


class ActorLoss(Protocol):
    def compute_actor_loss(
        self,
        actor: Callable[[Batch], tuple[torch.Tensor, torch.Tensor]],
        rollouts: RolloutBatch,
    ) -> ActorLossResults: ...


class ActorLossMixin:
    loss: torch.Tensor

    critic_metrics: CriticMetrics

    def metric_dict(self):
        return {
            **{
                "actor/" + field.name: getattr(self, field.name)
                for field in fields(self)
                if field.name != "critic_metrics"
            },
            **{"critic/" + k: v for k, v in self.critic_metrics._asdict().items()},
        }
