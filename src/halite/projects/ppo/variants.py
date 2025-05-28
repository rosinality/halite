from dataclasses import dataclass
from typing import Callable
import torch

from halite.projects.ppo import math_fn
from halite.projects.ppo.types import (
    ActorLoss,
    ActorLossMixin,
    Batch,
    CriticMetrics,
    Rollouts,
)


@torch.no_grad()
def compute_critic_metrics(rollouts: Rollouts, response_mask: torch.Tensor):
    advantage_mean = math_fn.masked_mean(rollouts.advantages, response_mask)
    advantage_var = math_fn.masked_var(
        rollouts.advantages, advantage_mean, response_mask
    )
    return_mean = math_fn.masked_mean(rollouts.returns, response_mask)
    return_var = math_fn.masked_var(rollouts.returns, return_mean, response_mask)

    return CriticMetrics(
        rewards_mean=rollouts.rewards.sum(-1).mean(),
        advantage_mean=advantage_mean,
        advantage_var=advantage_var,
        return_mean=return_mean,
        return_var=return_var,
    )


@dataclass
class PPOActorLossResults(ActorLossMixin):
    loss: torch.Tensor
    pg_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    pg_clipfrac: torch.Tensor

    critic_metrics: CriticMetrics


class PPOActorLoss:
    def __init__(self, clip_low, clip_high, pg_loss_agg, pg_loss_max_tokens):
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.pg_loss_agg = pg_loss_agg
        self.pg_loss_max_tokens = pg_loss_max_tokens

    def compute_actor_loss(
        self,
        actor: Callable[[Batch], tuple[torch.Tensor, torch.Tensor]],
        rollouts: Rollouts,
    ) -> PPOActorLossResults:
        actor_out, entropy = actor(rollouts.batch)

        log_ratio = actor_out - rollouts.actor_log_probs

        ratio = torch.exp(log_ratio)

        pg_loss1 = -rollouts.advantages * ratio
        pg_loss2 = -rollouts.advantages * torch.clamp(
            ratio, min=1 - self.clip_low, max=1 + self.clip_high
        )
        pg_loss = torch.maximum(pg_loss1, pg_loss2)

        response_len = rollouts.batch.response_ids.shape[-1]
        response_mask = rollouts.batch.attention_mask[:, -response_len:]

        pg_loss = math_fn.aggregate_loss(
            pg_loss,
            response_mask,
            self.pg_loss_agg,
            self.pg_loss_max_tokens,
        )

        with torch.no_grad():
            pg_clipfrac = math_fn.masked_mean(
                (pg_loss2 > pg_loss1).to(torch.float32),
                response_mask,
            )
            approx_kl = math_fn.masked_mean((ratio - 1) - log_ratio, response_mask)

        return PPOActorLossResults(
            loss=pg_loss,
            pg_loss=pg_loss,
            pg_clipfrac=pg_clipfrac,
            entropy=entropy,
            approx_kl=approx_kl,
            critic_metrics=compute_critic_metrics(rollouts, response_mask),
        )


@dataclass
class PPOKLCovActorLossResults(ActorLossMixin):
    loss: torch.Tensor
    pg_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor

    critic_metrics: CriticMetrics


class PPOKLCovActorLoss:
    def __init__(
        self,
        pg_loss_agg,
        pg_loss_max_tokens,
        kl_coef,
        select_ratio,
    ):
        self.pg_loss_agg = pg_loss_agg
        self.pg_loss_max_tokens = pg_loss_max_tokens
        self.kl_coef = kl_coef
        self.select_ratio = select_ratio

    def compute_actor_loss(
        self,
        actor: Callable[[Batch], tuple[torch.Tensor, torch.Tensor]],
        rollouts: Rollouts,
    ) -> PPOKLCovActorLossResults:
        log_probs, entropy = actor(rollouts.batch)

        log_ratio = log_probs - rollouts.actor_log_probs

        ratio = torch.exp(log_ratio)

        response_len = rollouts.batch.response_ids.shape[-1]
        response_mask = rollouts.batch.attention_mask[:, -response_len:]

        pg_loss1 = -rollouts.advantages * ratio

        valid_mask = response_mask > 0
        valid_log_probs = log_probs[valid_mask].detach().reshape(-1).cpu()
        valid_advantages = rollouts.advantages[valid_mask].detach().reshape(-1).cpu()

        select_ratio = min(self.select_ratio, len(valid_advantages))

        if select_ratio != 0:
            cov = (valid_advantages - valid_advantages.mean()) * (
                valid_log_probs - valid_log_probs.mean()
            )
            selected_n = max(1, int(len(cov) * select_ratio))
            large_cov_indices = torch.topk(cov, selected_n, largest=True).indices

            if len(large_cov_indices) != 0:
                abs_kl = log_ratio.abs()
                pg_loss1_kl = pg_loss1 + self.kl_coef * abs_kl

                valid_indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=True)[0]
                large_cov_indices = valid_indices[large_cov_indices]
                pg_loss1[
                    large_cov_indices // rollouts.advantages.shape[1],
                    large_cov_indices % rollouts.advantages.shape[1],
                ] = pg_loss1_kl[
                    large_cov_indices // rollouts.advantages.shape[1],
                    large_cov_indices % rollouts.advantages.shape[1],
                ]

        pg_loss = math_fn.aggregate_loss(
            pg_loss1,
            response_mask,
            self.pg_loss_agg,
            self.pg_loss_max_tokens,
        )

        with torch.no_grad():
            approx_kl = math_fn.masked_mean((ratio - 1) - log_ratio, response_mask)

        return PPOKLCovActorLossResults(
            loss=pg_loss,
            pg_loss=pg_loss,
            entropy=entropy,
            approx_kl=approx_kl,
            critic_metrics=compute_critic_metrics(rollouts, response_mask),
        )


@dataclass
class PPOAdaptiveEntropyActorLossResults(ActorLossMixin):
    loss: torch.Tensor
    pg_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    pg_clipfrac: torch.Tensor
    entropy_coef: float
    critic_metrics: CriticMetrics


class PPOAdaptiveEntropyActorLoss:
    def __init__(
        self,
        clip_low,
        clip_high,
        pg_loss_agg,
        pg_loss_max_tokens,
        init_entropy_coef,
        min_entropy_coef,
        max_entropy_coef,
        entropy_coef_update_size,
        target_entropy,
    ):
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.pg_loss_agg = pg_loss_agg
        self.pg_loss_max_tokens = pg_loss_max_tokens

        self.entropy_coef = init_entropy_coef
        self.min_entropy_coef = min_entropy_coef
        self.max_entropy_coef = max_entropy_coef
        self.entropy_coef_update_size = entropy_coef_update_size
        self.target_entropy = target_entropy

    @torch.no_grad()
    def update_entropy_coef(self, entropy):
        if entropy < self.target_entropy:
            self.entropy_coef += self.entropy_coef_update_size

        else:
            self.entropy_coef -= self.entropy_coef_update_size

        self.entropy_coef = min(
            max(self.entropy_coef, self.min_entropy_coef), self.max_entropy_coef
        )

    def compute_actor_loss(
        self,
        actor: Callable[[Batch], tuple[torch.Tensor, torch.Tensor]],
        rollouts: Rollouts,
    ) -> PPOAdaptiveEntropyActorLossResults:
        actor_out, entropy = actor(rollouts.batch)

        log_ratio = actor_out - rollouts.actor_log_probs

        ratio = torch.exp(log_ratio)

        pg_loss1 = -rollouts.advantages * ratio
        pg_loss2 = -rollouts.advantages * torch.clamp(
            ratio, min=1 - self.clip_low, max=1 + self.clip_high
        )
        pg_loss = torch.maximum(pg_loss1, pg_loss2)

        response_len = rollouts.batch.response_ids.shape[-1]
        response_mask = rollouts.batch.attention_mask[:, -response_len:]

        pg_loss = math_fn.aggregate_loss(
            pg_loss,
            response_mask,
            self.pg_loss_agg,
            self.pg_loss_max_tokens,
        )

        entropy = math_fn.masked_mean(entropy, response_mask)
        self.update_entropy_coef(entropy.item())

        loss = pg_loss  # - self.entropy_coef * entropy

        with torch.no_grad():
            pg_clipfrac = math_fn.masked_mean(
                (pg_loss2 > pg_loss1).to(torch.float32),
                response_mask,
            )
            approx_kl = math_fn.masked_mean((ratio - 1) - log_ratio, response_mask)

        return PPOAdaptiveEntropyActorLossResults(
            loss=loss,
            pg_loss=pg_loss.detach(),
            pg_clipfrac=pg_clipfrac,
            entropy=entropy.detach(),
            approx_kl=approx_kl,
            entropy_coef=self.entropy_coef,
            critic_metrics=compute_critic_metrics(rollouts, response_mask),
        )
