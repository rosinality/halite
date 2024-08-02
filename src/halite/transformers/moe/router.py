import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from halite.transformers.initialize import init_weights
from halite.transformers.moe import ops


def get_capacity(
    n_tokens: int, n_experts: int, capacity_factor: float, min_capacity=None
):
    capacity = math.ceil((n_tokens / n_experts) * capacity_factor)

    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity

    return capacity


def top_k_softmax_with_capacity(
    logits: torch.Tensor,
    top_k: int,
    capacity_factor: float | None = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
    pre_softmax: bool = False,
    deterministic: bool = False,
):
    n_tokens = logits.shape[0]
    n_experts = logits.shape[1]

    if pre_softmax:
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        probs, top_indices = torch.topk(scores, k=top_k, dim=1)

    else:
        assert top_k != 1

        scores, top_indices = torch.topk(logits, k=top_k, dim=1)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)

    if capacity_factor is None:
        if deterministic:
            tokens_per_expert = ops.bincount(top_indices.view(-1), minlength=n_experts)

        else:
            tokens_per_expert = torch.histc(
                top_indices, bins=n_experts, min=0, max=n_experts
            )

        return probs, top_indices, tokens_per_expert

    expert_capacity = get_capacity(n_tokens * top_k, n_experts, capacity_factor)
    top_k_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
    top_k_mask = torch.zeros_like(logits).scatter(1, top_indices, 1)

    capacity_probs, capacity_indices = torch.topk(
        top_k_masked_gates, k=expert_capacity, dim=0, sorted=False
    )
    capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)

    if drop_policy == "position":
        capacity_probs = torch.gather(top_k_masked_gates, 0, capacity_indices)

    if pad_to_capacity:
        final_probs = capacity_probs.T.contiguous()
        final_indices = capacity_indices.T.contiguous()
        tokens_per_expert_before_capacity = top_k_mask.sum(0)

    else:
        final_mask = torch.logical_and(top_k_mask, capacity_mask)
        drop_mask = torch.logical_not(final_mask)
        exceed_mask = torch.gather(drop_mask, 1, top_indices)
        final_probs = probs * torch.logical_not(exceed_mask)
        final_indices = top_indices.clone().masked_fill_(
            exceed_mask, torch.iinfo(torch.int64).max
        )
        tokens_per_expert_before_capacity = top_k_mask.sum(0)

    return final_probs, final_indices, tokens_per_expert_before_capacity


class TopKRouter(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_experts: int,
        top_k: int,
        gate_init: Callable = None,
        z_loss: float = 0,
        load_balance_loss: float = 0,
        capacity_factor: float | None = None,
        pad_to_capacity: bool = False,
        drop_policy: str = "probs",
        pre_softmax: bool = False,
        deterministic: bool = False,
    ):
        super().__init__()

        self.n_experts = n_experts
        self.gate = nn.Linear(in_features, n_experts, bias=False)
        self.gate_init = gate_init
        self.top_k = top_k
        self.z_loss = z_loss
        self.load_balance_loss = load_balance_loss

        self.capacity_factor = capacity_factor
        self.pad_to_capacity = pad_to_capacity
        self.drop_policy = drop_policy
        self.pre_softmax = pre_softmax
        self.deterministic = deterministic

    def init_weights(self):
        init_weights(self.gate, self.gate_init)

    def calc_z_loss(self, logits):
        z_loss = None

        if self.z_loss > 0 and self.training:
            z_loss = self.z_loss * torch.logsumexp(logits, dim=-1).square().mean()

        return z_loss

    def calc_load_balance_loss(self, logits, n_local_tokens_per_expert, activation):
        load_balance_loss = None

        if self.load_balance_loss > 0 and self.training:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)

            n_subseq = 1
            n_tokens = probs.shape[0] * n_subseq
            n_experts = probs.shape[1]

            probs_per_expert = probs.sum(0)
            n_local_tokens_per_expert = n_local_tokens_per_expert.to(probs_per_expert)
            load_balance_loss = (
                self.load_balance_loss
                * n_experts
                * (
                    F.normalize(probs_per_expert, p=1, dim=0)
                    * F.normalize(n_local_tokens_per_expert, p=1, dim=0)
                ).sum()
            )
            # load_balance_loss = torch.sum(
            #     probs_per_expert * n_local_tokens_per_expert.to(probs_per_expert)
            # ) * (
            #     n_experts * self.load_balance_loss / (n_tokens * n_tokens * self.top_k)
            # )

        return load_balance_loss

    def forward(self, input):
        aux_losses = {}
        logits = self.gate(input)

        logits = logits.view(-1, self.n_experts)

        z_loss = self.calc_z_loss(logits)

        if z_loss is not None:
            aux_losses["router-z-loss"] = z_loss

        probs, indices, tokens_per_expert = top_k_softmax_with_capacity(
            logits,
            self.top_k,
            self.capacity_factor,
            self.pad_to_capacity,
            self.drop_policy,
            self.pre_softmax,
            self.deterministic,
        )

        load_balance_loss = self.calc_load_balance_loss(
            logits, tokens_per_expert, activation=probs
        )
        if load_balance_loss is not None:
            aux_losses["load-balance-loss"] = load_balance_loss

        return probs, indices, aux_losses
