import torch
from torch import nn

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
)

from halite.nn.fused_linear_cross_entropy import fused_linear_cross_entropy
from halite.transformers.initialize import init_weights


class FusedLinearCrossEntropy(nn.Module):
    def __init__(
        self,
        linear,
        linear_init=None,
        ignore_index=-100,
        z_loss=0,
    ):
        super().__init__()

        self.linear = linear

        self.linear_init = linear_init
        self.ignore_index = ignore_index
        self.z_loss = z_loss

    def init_weights(self):
        init_weights(self.linear, self.linear_init)

    def forward(self, input, target):
        use_z_loss = self.z_loss > 0

        input = input.view(-1, input.shape[-1])
        target = target.view(-1)

        loss, z_loss = fused_linear_cross_entropy(
            input,
            self.linear.weight,
            target,
            self.linear.bias,
            ignore_index=self.ignore_index,
            lse_square_scale=self.z_loss,
            return_z_loss=use_z_loss,
        )

        loss_dict = {"cross entropy": loss.detach()}

        if use_z_loss:
            z_loss = z_loss.detach().mean()
            loss_dict["z-loss"] = z_loss
            loss_dict["cross entropy"] -= z_loss

        return loss, loss_dict


class VocabParallelLinear(nn.Module):
    def __init__(self, linear, linear_init=None, scale=None):
        super().__init__()

        self.linear = linear
        self.linear_init = linear_init

        self.scale = None
        if scale:
            self.scale = nn.Parameter(torch.ones(self.linear.out_features))

        self.tie_weight_key = "linear.weight"

    def init_weights(self):
        init_weights(self.linear, self.linear_init)

    def forward(self, input):
        return self.linear(input)

    def tie_weight(self, weight):
        self.linear.weight = weight

    @property
    def n_vocab(self):
        return self.linear.weight.shape[0]

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def parallelize_plan(self, **kwargs):
        loss_parallel = kwargs.get("loss_parallel", False)

        return {
            "linear": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
            )
        }


class SequenceParallelWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, input):
        return self.module(input)

    def parallelize_plan(self, **kwargs):
        return {"module": SequenceParallel()}


class FeedForward(nn.Module):
    def __init__(
        self,
        linear1,
        activation,
        linear2,
        dropout=0,
        linear1_init=None,
        linear2_init=None,
    ):
        super().__init__()

        self.linear1 = linear1
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear2

        self.linear1_init = linear1_init
        self.linear2_init = linear2_init

    def init_weights(self):
        init_weights(self.linear1, self.linear1_init)
        init_weights(self.linear2, self.linear2_init)

    def forward(self, input):
        out = self.linear1(input)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out

    def parallelize_plan(self, **kwargs):
        return {
            ".": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "linear1": ColwiseParallel(),
            "linear2": RowwiseParallel(
                output_layouts=Shard(1),
            ),
        }


class GatedFeedForward(nn.Module):
    def __init__(
        self,
        linear_proj,
        linear_gate,
        activation,
        linear_out,
        dropout=0,
        linear_proj_init=None,
        linear_gate_init=None,
        linear_out_init=None,
    ):
        super().__init__()

        self.linear_proj = linear_proj
        self.linear_gate = linear_gate
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear_out = linear_out

        self.linear_proj_init = linear_proj_init
        self.linear_gate_init = linear_gate_init
        self.linear_out_init = linear_out_init

    def init_weights(self):
        init_weights(self.linear_proj, self.linear_proj_init)
        init_weights(self.linear_gate, self.linear_gate_init)
        init_weights(self.linear_out, self.linear_out_init)

    def forward(self, input):
        proj = self.linear_proj(input)
        gate = self.activation(self.linear_gate(input))
        out = proj * gate
        out = self.dropout(out)

        out = self.linear_out(out)

        return out

    def parallelize_plan(self, **kwargs):
        return {
            ".": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "linear_proj": ColwiseParallel(),
            "linear_gate": ColwiseParallel(),
            "linear_out": RowwiseParallel(
                output_layouts=Shard(1),
            ),
        }
