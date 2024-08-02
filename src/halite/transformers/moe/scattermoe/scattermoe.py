import torch
from torch import nn
from scattermoe import kernels

from halite.transformers.initialize import init_weights
from halite.transformers.moe.scattermoe.ops import expert_boundaries, scattered_experts


class ExpertLinear(nn.Module):
    def __init__(self, n_experts, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(n_experts, in_features, out_features))

    def forward(
        self,
        input: torch.Tensor,
        k: int,
        sorted_expert_indices: torch.Tensor,
        sorted_scattered_indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        gates: torch.Tensor | None = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ):
        return scattered_experts(
            inputs=input,
            expert_weights=self.weight,
            k=k,
            sorted_expert_idxs=sorted_expert_indices,
            sorted_scattered_idxs=sorted_scattered_indices,
            expert_offsets=expert_offsets,
            gates=gates,
            grouped_in=grouped_in,
            grouped_out=grouped_out,
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        linear1,
        activation,
        linear2,
        top_k,
        dropout=0,
        linear1_init=None,
        linear2_init=None,
    ):
        super().__init__()

        self.linear1 = linear1
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2
        self.n_experts = linear1.weight.shape[0]
        self.top_k = top_k

        self.linear1_init = linear1_init
        self.linear2_init = linear2_init

    def init_weights(self):
        init_weights(self.linear1, self.linear1_init)
        init_weights(self.linear2, self.linear2_init)

    def forward(
        self,
        input: torch.Tensor,
        expert_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ):
        input_shape = input.shape
        input = input.view(-1, input_shape[-1])

        with torch.no_grad():
            sorted_expert_indices, sorted_scattered_indices = (
                expert_indices.flatten().sort()
            )
            expert_offsets = expert_boundaries(sorted_expert_indices, self.n_experts)

        out = self.linear1(
            input,
            self.top_k,
            sorted_expert_indices,
            sorted_scattered_indices,
            expert_offsets,
            grouped_out=True,
        )
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(
            out,
            1,
            sorted_expert_indices,
            sorted_scattered_indices,
            expert_offsets,
            grouped_in=True,
            gates=expert_probs,
        )
        out = out.view(*input_shape[:-1], out.shape[-1])

        return out
