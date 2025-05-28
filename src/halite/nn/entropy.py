import torch

import triton
import triton.language as tl


def get_num_warps(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return num_warps


MAX_FUSED_SIZE = 65536


@triton.jit
def entropy_from_logits_kernel(
    output_ptr,
    lse_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    lse_row_stride,
    n_cols,
    logit_scale,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)

    base_col_offsets = tl.arange(0, BLOCK_SIZE)

    row_start_ptr = input_ptr + row_id * input_row_stride

    input_ptrs = row_start_ptr + base_col_offsets
    mask = base_col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
    row = row * logit_scale

    prev_m = tl.max(row, axis=0)
    shift_x = row - prev_m
    exp_x = tl.exp(shift_x)
    prev_d = tl.sum(exp_x, -1)
    softmax = exp_x / prev_d
    prev_A = tl.sum(softmax, -1)
    prev_s = softmax * (shift_x - tl.log(prev_d))
    prev_s = tl.where(mask, prev_s, 0)
    prev_s = tl.sum(prev_s, -1)

    for i in range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        col_offsets = i + base_col_offsets
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        row = row * logit_scale

        next_m = tl.maximum(prev_m, tl.max(row, -1))
        diff_m = prev_m - next_m
        exp_m_ratio = tl.exp(diff_m)
        exp_x = tl.exp(row - next_m)
        next_d = prev_d * exp_m_ratio + tl.sum(exp_x, -1)

        d_ratio = prev_d / next_d
        log_next_d = tl.log(next_d)
        exp_x_next_d = exp_x / next_d
        exp_m_d_ratio = exp_m_ratio * d_ratio

        A = tl.sum(exp_x_next_d, -1)
        A = prev_A * exp_m_d_ratio + A
        B = exp_m_d_ratio
        C = diff_m + tl.log(prev_d) - log_next_d

        D = exp_x_next_d * (row - next_m - log_next_d)
        D = tl.where(mask, D, 0)
        D = tl.sum(D, -1)

        next_s = (prev_s + prev_A * C) * B + D
        prev_d = next_d
        prev_m = next_m
        prev_s = next_s
        prev_A = A

    output_ptrs = output_ptr + row_id * output_row_stride

    prev_s = -prev_s
    tl.store(output_ptrs, prev_s)

    lse_ptrs = lse_ptr + row_id * lse_row_stride

    lse = tl.log(prev_d) + prev_m
    tl.store(lse_ptrs, lse)


def _entropy_from_logits_forward(x, logit_scale=1.0):
    shape = x.shape

    x = x.contiguous()

    x = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    num_warps = get_num_warps(BLOCK_SIZE)

    y = x.new_empty(n_rows, dtype=torch.float32)
    lse = torch.empty_like(y)

    entropy_from_logits_kernel[(n_rows,)](
        y,
        lse,
        x,
        x.stride(0),
        y.stride(0),
        lse.stride(0),
        n_cols,
        logit_scale=logit_scale,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    y = y.reshape(*shape[:-1])
    lse = lse.reshape(*shape[:-1])

    return y, lse


@torch.library.custom_op("halite::entropy_from_logits_forward", mutates_args={})
def _entropy_from_logits_forward_compileable(
    x: torch.Tensor, logit_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    return _entropy_from_logits_forward(x, logit_scale)


def entropy_from_logits_forward(
    x: torch.Tensor, logit_scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.compiler.is_compiling():
        return _entropy_from_logits_forward_compileable(x, logit_scale)

    return _entropy_from_logits_forward(x, logit_scale)


@triton.jit
def entropy_from_logits_backward_kernel(
    dx_ptr,
    dy_ptr,
    entropy_ptr,
    lse_ptr,
    input_ptr,
    dx_stride,
    dy_stride,
    entropy_stride,
    lse_stride,
    input_stride,
    n_cols,
    logit_scale,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int64)

    base_col_offsets = tl.arange(0, BLOCK_SIZE)

    dx_start_ptr = dx_ptr + row_id * dx_stride
    dy_start_ptr = dy_ptr + row_id * dy_stride
    entropy_start_ptr = entropy_ptr + row_id * entropy_stride
    input_start_ptr = input_ptr + row_id * input_stride
    lse_start_ptr = lse_ptr + row_id * lse_stride

    for i in range(0, n_cols, BLOCK_SIZE):
        col_offsets = i + base_col_offsets
        input_ptrs = input_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        lse = tl.load(lse_start_ptr)

        log_p = row * logit_scale - lse
        softmax = tl.exp(log_p)

        entropy = tl.load(entropy_start_ptr)
        dy = tl.load(dy_start_ptr)

        dx_val = dy * -logit_scale * softmax * (entropy + log_p)

        tl.store(dx_start_ptr + col_offsets, dx_val, mask=mask)


def _entropy_from_logits_backward(dy, entropy, lse, x, logit_scale=1.0):
    shape = x.shape

    dy = dy.contiguous()
    entropy = entropy.contiguous()
    lse = lse.contiguous()
    x = x.contiguous()

    dy = dy.reshape(-1)
    entropy = entropy.reshape(-1)
    lse = lse.reshape(-1)
    x = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x.shape

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    num_warps = get_num_warps(BLOCK_SIZE)

    dx = x.new_empty(n_rows, n_cols, dtype=torch.float32)

    entropy_from_logits_backward_kernel[(n_rows,)](
        dx,
        dy,
        entropy,
        lse,
        x,
        dx.stride(0),
        dy.stride(0),
        entropy.stride(0),
        lse.stride(0),
        x.stride(0),
        n_cols,
        logit_scale=logit_scale,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dx = dx.reshape(*shape)

    return dx


@torch.library.custom_op("halite::entropy_from_logits_backward", mutates_args={})
def _entropy_from_logits_backward_compileable(
    dy: torch.Tensor,
    entropy: torch.Tensor,
    lse: torch.Tensor,
    x: torch.Tensor,
    logit_scale: float = 1.0,
) -> torch.Tensor:
    return _entropy_from_logits_backward(dy, entropy, lse, x, logit_scale)


def entropy_from_logits_backward(
    dy: torch.Tensor,
    entropy: torch.Tensor,
    lse: torch.Tensor,
    x: torch.Tensor,
    logit_scale: float = 1.0,
) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return _entropy_from_logits_backward_compileable(
            dy, entropy, lse, x, logit_scale
        )

    return _entropy_from_logits_backward(dy, entropy, lse, x, logit_scale)


class EntropyFromLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, logit_scale=1.0):
        y, lse = entropy_from_logits_forward(logits, logit_scale)
        ctx.save_for_backward(y, logits, lse)
        ctx.logit_scale = logit_scale

        return y

    @staticmethod
    def backward(ctx, grad_output):
        entropy, logits, lse = ctx.saved_tensors
        ent_grad = entropy_from_logits_backward(
            grad_output, entropy, lse, logits, ctx.logit_scale
        )

        return ent_grad, None


def entropy_from_logits(logits, logit_scale=1.0):
    return EntropyFromLogits.apply(logits, logit_scale)
