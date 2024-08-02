from audioop import reverse
from dataclasses import dataclass

import torch
from torch import nn


class MoEGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.input_shape = input.shape
        ctx.indices = indices

        return torch.gather(input, 0, indices)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.new_zeros(ctx.input_shape)
        output.scatter_add_(0, ctx.indices, grad_output)

        return output, None


def moe_gather(input, indices):
    return MoEGather.apply(input, indices)


class MoEScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices, output_size):
        ctx.input_shape = input.shape
        ctx.indices = indices
        ctx.output_size = output_size

        if output_size is not None:
            output = input.new_zeros(output_size)

        else:
            output = torch.zeros_like(input)

        output.scatter_add_(0, indices, input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.gather(grad_output, 0, ctx.indices)

        return grad_input, None, None


def moe_scatter(input, indices, output_shape=None):
    return MoEScatter.apply(input, indices, output_shape)


class AlltoAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output_split_sizes, input_split_sizes, group):
        ctx.group = group
        ctx.output_splite_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = torch.distributed.get_world_size(group)

        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            output = torch.empty_like(input)

        else:
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.shape[1:])
            )

        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (
            AlltoAll.apply(
                *grad_output, ctx.input_split_sizes, ctx.output_split_sizes, ctx.group
            ),
            None,
            None,
            None,
        )


def all_to_all(input, output_split_sizes=None, input_split_sizes=None, group=None):
    return AlltoAll.apply(input, output_split_sizes, input_split_sizes, group)


def gather_along_first_dim_moe(input, group):
    world_size = torch.distributed.get_world_size(group)

    if world_size == 1:
        return input

    dim_size = list(input.size())
    dim_size[0] = dim_size[0] * world_size

    output = input.new_empty(dim_size)
    torch.distributed._all_gather_base(output, input.contiguous(), group=group)

    return output


def permute(tokens, indices, n_out_tokens: int | None = None, pad_mode: bool = False):
    if pad_mode:
        return permute_with_pad(tokens, indices)

    if indices.ndim == 1:
        indices = indices.unsqueeze(1)

    top_k = indices.shape[1]
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)

    if n_out_tokens is not None:
        sorted_indices = sorted_indices[:n_out_tokens]

    moe_gather_indices = (
        (sorted_indices // top_k).unsqueeze(1).expand(-1, tokens.shape[-1])
    )
    permuted_tokens = moe_gather(tokens, moe_gather_indices)

    return permuted_tokens, sorted_indices


def permute_with_pad(tokens, indices):
    permuted_tokens = tokens.index_select(0, indices.view(-1))

    return permuted_tokens, indices


def unpermute(
    permuted_tokens,
    sorted_indices,
    probs: torch.Tensor | None = None,
    pad_mode: bool = False,
    restore_shape: torch.Size = None,
):
    if pad_mode:
        return unpermute_with_pad(permuted_tokens, sorted_indices, probs, restore_shape)

    if probs is not None:
        n_unpermuted_tokens = probs.numel()
        top_k = probs.shape[1]

    else:
        n_unpermuted_tokens = permuted_tokens.shape[0]
        top_k = 1

    output_size = [n_unpermuted_tokens, permuted_tokens.shape[-1]]

    print("unpermute output size", output_size)

    moe_scatter_indices = sorted_indices.unsqueeze(1).expand(
        -1, permuted_tokens.shape[-1]
    )
    unpermuted_tokens = moe_scatter(permuted_tokens, moe_scatter_indices, output_size)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, top_k, permuted_tokens.shape[-1])

    print("unpermuted tokens", unpermuted_tokens)

    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)

    unpermuted_tokens = unpermuted_tokens.sum(1)

    return unpermuted_tokens


def unpermute_with_pad(permuted_tokens, sorted_indices, probs, restore_shape):
    probs = probs.view(-1).unsqueeze(-1)
    indices = sorted_indices.view(-1, 1).expand(-1, permuted_tokens.shape[1])
    combined_output = probs * permuted_tokens
    empty_tokens = combined_output.new_zeros(restore_shape)
    unpermuted_tokens = torch.scatter_add(empty_tokens, 0, indices, combined_output)

    return unpermuted_tokens


def sort_chunks_by_indices(input, split_sizes, sorted_indices):
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_indices], dim=0)

    return output


@dataclass
class PermuteState:
    n_tokens_per_expert: torch.Tensor
    n_out_tokens: int | None = None
    output_splits: torch.Tensor | None = None
    input_splits: torch.Tensor | None = None
    n_global_tokens_per_local_expert: torch.Tensor | None = None
    input_shape: torch.Size | None = None
    reverse_permute_indices: torch.Tensor | None = None
    input_shape_before_permute: torch.Size | None = None
    cuda_sync_point: str = "no_sync"


class AlltoAllTokenDispatcher(nn.Module):
    def __init__(
        self,
        n_experts: int,
        capacity_factor: float | None = None,
        pad_to_capacity: bool = False,
        deterministic: bool = False,
    ):
        super().__init__()

        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.pad_to_capacity = pad_to_capacity
        self.deterministic = deterministic

    def parallelize(self, ep_group):
        self.ep_group = ep_group
        self.ep_size = torch.distributed.get_world_size(ep_group)
        self.n_local_experts = self.n_experts // self.ep_size
        input_chunk_indices = torch.arange(self.n_experts)
        self.sort_input_by_local_experts = (
            input_chunk_indices.reshape(-1, self.n_local_experts).T.ravel().tolist()
        )
        self.restore_output_by_local_experts = (
            input_chunk_indices.reshape(self.n_local_experts, -1).T.ravel().tolist()
        )
        self.local_expert_indices = list(range(self.n_local_experts))

    def prepare(self, indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        n_out_tokens = None
        output_splits = None
        input_splits = None
        n_global_tokens_per_local_expert_cpu = None

        if self.pad_to_capacity:
            capacity = probs.shape[1]
            n_tokens_per_local_expert = torch.full(
                (self.n_local_experts,), capacity * self.ep_size, dtype=torch.int64
            )
            n_global_tokens_per_local_expert_cpu = torch.full(
                (self.n_experts,), capacity, dtype=torch.int64
            )

            return PermuteState(
                n_tokens_per_expert=n_tokens_per_local_expert,
                n_out_tokens=n_out_tokens,
                output_splits=output_splits,
                input_splits=input_splits,
                n_global_tokens_per_local_expert=n_global_tokens_per_local_expert_cpu,
            )

        if self.deterministic:
            n_local_tokens_per_expert = torch.bincount(
                indices.view(-1), minlength=self.n_experts
            )

        else:
            n_local_tokens_per_expert = torch.histc(
                indices, bins=self.n_experts, min=0, max=self.n_experts
            )

        if self.capacity_factor is not None:
            n_out_tokens = n_local_tokens_per_expert.sum().to("cpu", non_blocking=True)
            cuda_sync_point = "before_permutation_1"
            print("n_out_tokens", n_out_tokens, indices)

        n_global_tokens_per_local_expert = n_local_tokens_per_expert.reshape(
            self.n_experts
        )
        n_tokens_per_expert = n_local_tokens_per_expert.to("cpu", non_blocking=True)

        if self.ep_size > 1:
            input_splits = (
                n_local_tokens_per_expert.reshape(self.ep_size, self.n_local_experts)
                .sum(1)
                .to("cpu", non_blocking=True)
                .numpy()
            )
            n_global_tokens_per_expert = (
                gather_along_first_dim_moe(n_local_tokens_per_expert, self.ep_group)
                .reshape(self.ep_size, 1, self.n_experts)
                .transpose(0, 1)
            )
            n_global_tokens_per_local_expert = n_global_tokens_per_expert[
                :, :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()
            n_global_tokens_per_rank = n_global_tokens_per_local_expert.sum(2)
            output_splits = (
                n_global_tokens_per_rank[0].to("cpu", non_blocking=True).numpy()
            )
            n_tokens_per_local_expert = n_global_tokens_per_local_expert.sum((0, 1)).to(
                "cpu", non_blocking=True
            )

        if self.n_local_experts > 1:
            n_global_tokens_per_local_expert_cpu = (
                n_global_tokens_per_local_expert.view(-1, self.n_local_experts).to(
                    "cpu", non_blocking=True
                )
            )

        return PermuteState(
            n_tokens_per_expert=n_tokens_per_expert,
            n_out_tokens=n_out_tokens,
            output_splits=output_splits,
            input_splits=input_splits,
            n_global_tokens_per_local_expert=n_global_tokens_per_local_expert_cpu,
            cuda_sync_point=cuda_sync_point,
        )

    def permute(
        self,
        input: torch.Tensor,
        probs: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = input.shape
        input = input.view(-1, input.shape[-1])
        input_shape_before_permute = input.shape

        permute_state = self.prepare(indices, probs)

        if permute_state.cuda_sync_point == "before_permutation_1":
            torch.cuda.current_stream().synchronize()

        permuted_local_input_tokens, reverse_permute_indices = permute(
            input, indices, permute_state.n_out_tokens, self.pad_to_capacity
        )
        print("permuted local input tokens", permuted_local_input_tokens)
        global_input_tokens = all_to_all(
            permuted_local_input_tokens,
            permute_state.output_splits,
            permute_state.input_splits,
            self.ep_group,
        )

        print("splits", permute_state.output_splits, permute_state.input_splits)
        print("all to all global input tokens", global_input_tokens)

        if self.n_local_experts > 1:
            global_input_tokens = sort_chunks_by_indices(
                global_input_tokens,
                permute_state.n_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
            )

        print("global input tokens", global_input_tokens)

        permute_state.input_shape = input_shape
        permute_state.reverse_permute_indices = reverse_permute_indices
        permute_state.input_shape_before_permute = input_shape_before_permute

        return global_input_tokens, permute_state

    def unpermute(self, input, probs, permute_state: PermuteState) -> torch.Tensor:
        if self.n_local_experts > 1:
            n_global_tokens_per_local_expert = (
                permute_state.n_global_tokens_per_local_expert
            )

            if n_global_tokens_per_local_expert.ndim > 1:
                n_global_tokens_per_local_expert = n_global_tokens_per_local_expert.T

            input = sort_chunks_by_indices(
                input,
                n_global_tokens_per_local_expert.ravel(),
                self.restore_output_by_local_experts,
            )

        permuted_local_input_tokens = all_to_all(
            input,
            permute_state.input_splits,
            permute_state.output_splits,
            self.ep_group,
        )

        output = unpermute(
            permuted_local_input_tokens,
            permute_state.reverse_permute_indices,
            probs=probs,
            pad_mode=self.pad_to_capacity,
            restore_shape=permute_state.input_shape_before_permute,
        )

        output = output.view(permute_state.input_shape)

        return output
