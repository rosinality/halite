import torch
import triton
import triton.language as tl

from halite.transformers.moe.ops import bincount
from halite.transformers.moe.scattermoe.kernels import (
    group_triton_kernel,
    groupXtY_triton_kernel,
    scatter2scatter_triton_kernel,
)


BLOCK_M = 128
torch._dynamo.config.capture_scalar_outputs = True


def _scatter2scatter(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    out: torch.Tensor,
    FAN_OUT: int,
    x_grouped: bool = False,
    y_grouped: bool = False,
) -> None:
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * FAN_OUT
    assert out.size(0) == sorted_expert_idxs.size(0)
    assert out.size(1) == W.size(-1)

    grid = lambda meta: (
        triton.cdiv(sorted_expert_idxs.size(0), meta["BLOCK_M"])
        * triton.cdiv(meta["N"], meta["BLOCK_N"]),
    )

    with torch.device(X.device):
        scatter2scatter_triton_kernel[grid](
            # X_ptr, stride_xm, stride_xk,
            X,
            X.stride(0),
            X.stride(1),
            # W_ptr, stride_we, stride_wk, stride_wn,
            W,
            W.stride(0),
            W.stride(1),
            W.stride(2),
            # Y_ptr, stride_ym, stride_yn,
            out,
            out.stride(0),
            out.stride(1),
            grouped_idx_ptr=sorted_scattered_idxs,
            expert_idxs_ptr=sorted_expert_idxs,
            FAN_OUT=FAN_OUT,
            M=X.size(0),
            K=X.size(1),
            N=out.size(1),
            E=W.size(0),
            BLOCK_M=BLOCK_M,
            ACC_TYPE=tl.float32,
            allow_tf32=torch.backends.cudnn.allow_tf32,
            x_grouped=x_grouped,
            y_grouped=y_grouped,
        )


# custom op is needed because of https://github.com/pytorch/pytorch/issues/136394
@torch.library.custom_op("halite::scatter2scatter", mutates_args={"out"})
def _scatter2scatter_compileable(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    out: torch.Tensor,
    FAN_OUT: int,
    x_grouped: bool = False,
    y_grouped: bool = False,
) -> None:
    _scatter2scatter(
        X=X,
        W=W,
        sorted_expert_idxs=sorted_expert_idxs,
        sorted_scattered_idxs=sorted_scattered_idxs,
        out=out,
        FAN_OUT=FAN_OUT,
        x_grouped=x_grouped,
        y_grouped=y_grouped,
    )


def scatter2scatter(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    out: torch.Tensor,
    FAN_OUT: int,
    x_grouped: bool = False,
    y_grouped: bool = False,
) -> None:
    if torch.compiler.is_compiling():
        _scatter2scatter_compileable(
            X=X,
            W=W,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=out,
            FAN_OUT=FAN_OUT,
            x_grouped=x_grouped,
            y_grouped=y_grouped,
        )
    else:
        _scatter2scatter(
            X=X,
            W=W,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=out,
            FAN_OUT=FAN_OUT,
            x_grouped=x_grouped,
            y_grouped=y_grouped,
        )


def _group_bwd_W(
    DY: torch.Tensor,
    X: torch.Tensor,
    expert_offsets: torch.Tensor,
    DW: torch.Tensor,
    E: int,
) -> None:
    grid = lambda meta: (
        E * triton.cdiv(meta["K"], meta["BLOCK_K"]),
        triton.cdiv(meta["N"], meta["BLOCK_N"]),
    )

    with torch.device(X.device):
        groupXtY_triton_kernel[grid](
            # DY_ptr, stride_dym, stride_dyk,
            DY,
            DY.stride(0),
            DY.stride(1),
            # X_ptr, stride_xm, stride_xn,
            X,
            X.stride(0),
            X.stride(1),
            # DW_ptr, stride_dwe, stride_dwk, stride_dwn,
            DW,
            DW.stride(0),
            DW.stride(1),
            DW.stride(2),
            # expert_offsets_ptr,
            expert_offsets,
            # K: tl.constexpr, N: tl.constexpr,
            N=DY.size(-1),
            K=X.size(-1),
            # ACC_TYPE: tl.constexpr,
            ACC_TYPE=tl.float32,
            allow_tf32=torch.backends.cudnn.allow_tf32,
        )


# custom op is needed because of https://github.com/pytorch/pytorch/issues/136394
@torch.library.custom_op("halite::group_bwd_W", mutates_args={"DW"})
def _group_bwd_W_compileable(
    DY: torch.Tensor,
    X: torch.Tensor,
    expert_offsets: torch.Tensor,
    DW: torch.Tensor,
    E: int,
) -> None:
    _group_bwd_W(DY=DY, X=X, expert_offsets=expert_offsets, DW=DW, E=E)


def group_bwd_W(
    DY: torch.Tensor,
    X: torch.Tensor,
    expert_offsets: torch.Tensor,
    DW: torch.Tensor,
    E: int,
) -> None:
    if torch.compiler.is_compiling():
        _group_bwd_W_compileable(DY=DY, X=X, expert_offsets=expert_offsets, DW=DW, E=E)
    else:
        _group_bwd_W(DY=DY, X=X, expert_offsets=expert_offsets, DW=DW, E=E)


def _group(
    A: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    out: torch.Tensor,
    coeff: torch.Tensor | None = None,
    fan_out: int = 1,
) -> None:
    N = sorted_expert_idxs.size(0)
    K = A.size(1)
    assert A.size(0) * fan_out == N

    grid = lambda meta: (triton.cdiv(meta["N"], meta["BLOCK_N"]),)

    with torch.device(A.device):
        group_triton_kernel[grid](
            # A_ptr, stride_an, stride_ai,
            A,
            A.stride(0),
            A.stride(1),
            coeff is not None,
            coeff,
            fan_out,
            # Y_ptr, stride_yn, stride_yk,
            out,
            out.stride(0),
            out.stride(1),
            # grouped_idx_ptr,
            sorted_expert_idxs,
            # N: tl.constexpr, K: tl.constexpr,
            N,
            K,
        )


# custom op is needed because of https://github.com/pytorch/pytorch/issues/136394
@torch.library.custom_op("halite::group", mutates_args={"out"})
def _group_compileable(
    A: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    out: torch.Tensor,
    coeff: torch.Tensor | None = None,
    fan_out: int = 1,
) -> None:
    _group(
        A=A,
        sorted_expert_idxs=sorted_expert_idxs,
        out=out,
        coeff=coeff,
        fan_out=fan_out,
    )


def group(
    A: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    out: torch.Tensor,
    coeff: torch.Tensor | None = None,
    fan_out: int = 1,
) -> None:
    if torch.compiler.is_compiling():
        _group_compileable(
            A=A,
            sorted_expert_idxs=sorted_expert_idxs,
            out=out,
            coeff=coeff,
            fan_out=fan_out,
        )
    else:
        _group(
            A=A,
            sorted_expert_idxs=sorted_expert_idxs,
            out=out,
            coeff=coeff,
            fan_out=fan_out,
        )


BLOCK_M = 128
torch._dynamo.config.capture_scalar_outputs = True


def expert_boundaries(sorted_experts_idxs: torch.Tensor, k: int) -> torch.Tensor:
    # there is an overhead of launching a custom op so we only use the custom op when compiling
    if torch.compiler.is_compiling():
        expert_counts = bincount(sorted_experts_idxs, k)
    else:
        expert_counts = sorted_experts_idxs.bincount(minlength=k)
    expert_boundaries_end = expert_counts.cumsum(-1)
    return expert_boundaries_end


class _ScatteredExperts(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        output = torch.empty(
            sorted_expert_idxs.size(0),
            expert_weights.size(-1),
            device=x.device,
            dtype=x.dtype,
        )

        scatter2scatter(
            X=x,
            W=expert_weights,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=output,
            FAN_OUT=k,
            x_grouped=grouped_in,
            y_grouped=grouped_out,
        )

        if gates is None:
            output_expanded = None
        else:
            output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
            output = torch.bmm(gates.unsqueeze(1), output_expanded).squeeze(1)

        ctx.save_for_backward(
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            gates,
            output_expanded,
        )

        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k

        return output

    @staticmethod
    def backward(ctx, grad_out):
        (
            x,
            expert_weights,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            gates,
            output_expanded,
        ) = ctx.saved_tensors
        k = ctx.k
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out

        if gates is None:
            d_gates = None
            gates_flat = None
            gate_fan = 1
            grouped_grad_out = None
        else:
            # calculate gates gradient
            d_gates = torch.bmm(output_expanded, grad_out.unsqueeze(2)).squeeze(-1)
            gates_flat = gates.flatten()
            gate_fan = gates.size(1)
            # print("expanded and grouping")
            grouped_grad_out = output_expanded.flatten(
                0, 1
            )  # reuse expanded buffer later

        if grouped_out:
            grouped_grad_out = grad_out
        else:
            group(
                A=grad_out,
                sorted_expert_idxs=sorted_scattered_idxs,
                out=grouped_grad_out,
                coeff=gates_flat,
                fan_out=gate_fan,
            )

        if grouped_in:
            grouped_x = x
            d_expanded_input = torch.empty(
                sorted_expert_idxs.size(0),
                expert_weights.size(1),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            grouped_x = torch.empty(
                sorted_scattered_idxs.size(0), x.size(1), dtype=x.dtype, device=x.device
            )
            group(
                A=x,
                sorted_expert_idxs=sorted_scattered_idxs,
                out=grouped_x,
                fan_out=k,
            )

            d_expanded_input = grouped_x

        d_weights = torch.zeros_like(expert_weights)

        group_bwd_W(
            DY=grouped_grad_out,
            X=grouped_x,
            expert_offsets=expert_offsets,
            DW=d_weights,
            E=expert_weights.size(0),
        )

        scatter2scatter(
            X=grouped_grad_out,
            W=expert_weights.permute(0, 2, 1),
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            out=d_expanded_input,
            FAN_OUT=1,
            x_grouped=True,
            y_grouped=grouped_in,
        )

        if k == 1:
            d_input = d_expanded_input
        else:
            d_input = d_expanded_input.view(
                x.size(0), k, d_expanded_input.size(-1)
            ).sum(-2)

        # print("backward end.")
        return (
            # x, expert_weights, k,
            d_input,
            d_weights,
            None,
            # sorted_expert_idxs, sorted_scattered_idxs,
            None,
            None,
            # expert_offsets,
            None,
            # gates
            d_gates,
            None,
            None,
        )


def scattered_experts(
    inputs,
    expert_weights,
    k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    expert_offsets,
    gates=None,
    grouped_in=False,
    grouped_out=False,
):
    return _ScatteredExperts.apply(
        inputs,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gates,
        grouped_in,
        grouped_out,
    )
