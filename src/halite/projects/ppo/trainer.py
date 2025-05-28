from typing import Any, NamedTuple

import torch
from torch.nn import functional as F

from halite.projects.ppo import math_fn
from halite.projects.ppo.types import ActorLossResults, Batch, Rollouts
from halite.transformers.transformer import TransformerDecoderOutput


def build_batch_from_rollouts(rollouts, offset=0, pad_id=-1, device=None):
    ids = [sample["id"] for sample in rollouts.samples]
    prompts = [sample["input_ids"] for sample in rollouts.samples]
    responses = [sample["output_ids"] for sample in rollouts.samples]
    prompt_len = torch.tensor([len(p) for p in prompts], device=device)
    response_len = torch.tensor([len(r) for r in responses], device=device)

    prompt_max_len = max(len(p) for p in prompts)
    response_max_len = max(len(r) for r in responses)

    input_ids = torch.zeros(
        len(prompts),
        prompt_max_len + response_max_len,
        dtype=torch.int64,
        device=device,
    )

    attention_mask = torch.zeros_like(input_ids)
    position_ids = torch.zeros_like(input_ids)
    response_ids = input_ids.new_full((len(prompts), response_max_len), pad_id)

    for id, (prompt, response) in enumerate(zip(prompts, responses)):
        sample_prompt_len = len(prompt)
        sample_response_len = len(response)

        length = sample_prompt_len + sample_response_len
        response_tensor = torch.tensor(response, device=device)

        input_ids[id, prompt_max_len - sample_prompt_len : prompt_max_len] = (
            torch.tensor(prompt)
        )
        input_ids[id, prompt_max_len : prompt_max_len + sample_response_len] = (
            response_tensor
        )

        attention_mask[
            id,
            prompt_max_len - sample_prompt_len : prompt_max_len + sample_response_len,
        ] = 1
        position_ids[
            id,
            prompt_max_len - sample_prompt_len : prompt_max_len + sample_response_len,
        ] = torch.arange(length)

        response_ids[id, :sample_response_len] = response_tensor

    temperatures = [
        params.get("temperature", 1.0) for params in rollouts.sampling_params
    ]
    if all(t == temperatures[0] for t in temperatures):
        temperatures = temperatures[0]

    else:
        temperatures = torch.tensor(
            temperatures,
            device=device,
        )

    return Batch(
        input_ids,
        response_ids,
        attention_mask,
        position_ids,
        torch.tensor(ids, device=device),
        prompt_len,
        response_len,
        temperatures,
    )


def build_extract_response_mask(tensor, prompt_len, response_len, offset=0):
    prompt_len = torch.as_tensor(prompt_len, device=tensor.device)
    response_len = torch.as_tensor(response_len, device=tensor.device)

    arange = torch.arange(response_len.max().item(), device=tensor.device).unsqueeze(0)

    if offset != 0:
        prompt_len = prompt_len + offset
        response_len = response_len + offset

    index = prompt_len.unsqueeze(1) + arange
    mask = index <= (prompt_len + response_len).unsqueeze(1)
    index = index.masked_fill(~mask, 0).unsqueeze(-1)

    return index, mask


def apply_extract_response_mask(tensor, index, mask, pad_id=-1):
    values = torch.take_along_dim(tensor, index, dim=1)
    values = values.masked_fill(~mask.unsqueeze(-1), pad_id)

    return values


def extract_response(tensor, prompt_len, response_len, offset=0):
    index, mask = build_extract_response_mask(tensor, prompt_len, response_len, offset)

    values = tensor.gather(1, torch.tile(index.unsqueeze(-1), (1, 1, tensor.shape[-1])))
    values = values.masked_fill(~mask.unsqueeze(-1), 0)

    return values


def get_logits(output):
    if isinstance(output, TransformerDecoderOutput):
        return output.logits

    return output


def log_probs_from_logits(logits, labels, ignore_index, temperature=1.0):
    if isinstance(temperature, torch.Tensor) or temperature != 1.0:
        logits = logits / temperature.reshape(-1, 1, 1)

    return -F.cross_entropy(
        logits.permute(0, 2, 1), labels, reduction="none", ignore_index=ignore_index
    )


def entropy_from_logits(logits, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature.reshape(-1, 1, 1)

    return -torch.sum(torch.softmax(logits, -1) * torch.log_softmax(logits, -1), -1)


def compute_approx_kl(log_probs, ref_log_probs):
    log_ratio = log_probs - ref_log_probs
    approx_kl = (torch.exp(log_ratio) - 1) - log_ratio

    return approx_kl


class KLPenalty:
    def __init__(self, kl_coef, method="approx"):
        self.kl_coef = kl_coef
        self.method = method

    def __call__(self, rewards, actor_out, ref_out):
        if self.method == "approx":
            kl = compute_approx_kl(actor_out, ref_out)
        else:
            kl = actor_out - ref_out

        return rewards - self.kl_coef * kl


@torch.no_grad()
def compute_ppo_advantage(
    rewards, mask, values, gamma, lam, reward_whitening=True, **kwargs
):
    """
    Generalized Advantage Estimation
    You can refer to https://arxiv.org/abs/1506.02438
    """

    if reward_whitening:
        rewards = math_fn.whitening(rewards, shift_mean=False, mask=mask)

    last_gae = 0
    advantages_reversed = []
    gen_length = rewards.shape[1]

    values_unbind = values.unbind(1)
    rewards_unbind = rewards.unbind(1)

    for t in reversed(range(gen_length)):
        next_values = values_unbind[t + 1] if t < gen_length - 1 else 0.0
        delta = rewards_unbind[t] + gamma * next_values - values_unbind[t]
        last_gae = delta + gamma * lam * last_gae
        advantages_reversed.append(last_gae)

    advantages = torch.stack(advantages_reversed[::-1], 1)
    returns = advantages + values

    advantages = math_fn.whitening(advantages, shift_mean=True, mask=mask)

    return advantages, returns


@torch.no_grad()
def compute_grpo_advantage(
    rewards, mask, group_ids, std_normalize=True, eps=1e-6, **kwargs
):
    rewards = rewards.sum(-1)

    correction = 1

    unique_ids, idx = group_ids.unique(sorted=True, return_inverse=True)
    count = torch.bincount(idx, minlength=unique_ids.numel())
    mean = torch.bincount(idx, weights=rewards) / count
    mean[count == 1] = 0
    advantage = rewards - mean[idx]

    if std_normalize:
        sq_diff = advantage.square()
        var = torch.bincount(idx, weights=sq_diff) / (count - correction)
        std = var.sqrt()
        std[count == 1] = 1
        advantage = advantage / (std[idx] + eps)

    advantage = advantage.unsqueeze(-1) * mask

    return advantage, advantage


class PPOTrainer:
    def __init__(
        self,
        actor,
        advantage_fn,
        actor_loss,
        ref=None,
        penalty_fn=None,
        critic=None,
        log_probs_batch_size=None,
        device=None,
    ):
        self.actor = actor
        self.advantage_fn = advantage_fn
        self.ref = ref
        self.penalty_fn = penalty_fn
        self.critic = critic

        self.log_probs_batch_size = log_probs_batch_size

        self.actor_loss = actor_loss

        self.device = device
        self.pad_id = -1

    @torch.no_grad()
    def compute_advantage(self, rollouts):
        batch = build_batch_from_rollouts(
            rollouts, offset=-1, pad_id=self.pad_id, device=self.device
        )

        batches = [batch]
        if self.log_probs_batch_size is not None:
            batches = batch.split(self.log_probs_batch_size)

        actor_outputs = []
        actor_entropies = []
        ref_outputs = []
        critic_outputs = []
        for microbatch in batches:
            actor_out, actor_entropy, ref_out, critic_out = self.compute_log_probs(
                microbatch
            )

            actor_outputs.append(actor_out)
            actor_entropies.append(actor_entropy)

            if ref_out is not None:
                ref_outputs.append(ref_out)

            if critic_out is not None:
                critic_outputs.append(critic_out)

        actor_out = torch.cat(actor_outputs, 0)
        actor_entropy = torch.cat(actor_entropies, 0)

        ref_out = None
        if len(ref_outputs) > 0:
            ref_out = torch.cat(ref_outputs, 0)

        critic_out = None
        if len(critic_outputs) > 0:
            critic_out = torch.cat(critic_outputs, 0)

        rewards = torch.as_tensor(rollouts.rewards, device=self.device)
        if self.penalty_fn is not None:
            rewards = self.penalty_fn(rewards, actor_out, ref_out)

        response_len = batch.response_ids.shape[-1]
        response_mask = batch.attention_mask[:, -response_len:]

        advantages, returns = self.advantage_fn(
            rewards,
            response_mask,
            values=critic_out,
            group_ids=batch.ids,
        )

        rollouts_orig = {**rollouts._asdict(), "rewards": rewards}

        return Rollouts(
            **rollouts_orig,
            batch=batch,
            advantages=advantages,
            returns=returns,
            actor_log_probs=actor_out,
            actor_entropy=actor_entropy,
        )

    def compute_log_probs(self, batch):
        actor_out, actor_entropy = self.actor(batch)

        ref_out = None
        if self.ref is not None:
            ref_out, _ = self.ref(batch)

        critic_out = None
        if self.critic is not None:
            critic_out = self.critic(batch)

        return actor_out, actor_entropy, ref_out, critic_out

    def compute_actor_loss(self, rollouts: Rollouts) -> ActorLossResults:
        return self.actor_loss.compute_actor_loss(self.actor, rollouts)

    def compute_critic_loss(self, rollouts):
        pass
