import torch
from torch import distributed as dist
from torch.nn import functional as F


def masked_mean(values, masks):
    return torch.sum(values * masks) / masks.sum()


def masked_var(values, mean, masks):
    return ((values - mean) * masks).pow(2).sum() / masks.sum()


def dist_var_mean(tensor, group, mask=None, n_mask=None):
    if mask is not None:
        tensor_sum = torch.sum(tensor * mask)
        n_elem = n_mask

    else:
        tensor_sum = tensor.sum()
        n_elem = tensor.numel()

    sum_count = torch.tensor([tensor_sum, n_elem], device=tensor.device)
    dist.all_reduce(sum_count, dist.ReduceOp.SUM, group=group)
    global_sum, count = sum_count
    global_mean = global_sum / count

    if mask is not None:
        sum_var = torch.sum(((tensor - global_mean) * mask) ** 2)

    else:
        sum_var = torch.sum((tensor - global_mean) ** 2)

    dist.all_reduce(sum_var, dist.ReduceOp.SUM, group=group)
    global_var = sum_var / count

    return global_var, global_mean


def whitening(
    tensor: torch.Tensor,
    shift_mean: bool = True,
    eps: float = 1e-8,
    mask: torch.Tensor | None = None,
):
    if mask is not None:
        n_mask = mask.sum()

        mean = masked_mean(tensor, mask, n_mask)
        var = masked_var(tensor, mean, mask, n_mask)

    else:
        try:
            var, mean = torch.var_mean(tensor, correction=0)

        except TypeError:
            var, mean = torch.var_mean(tensor, unbiased=False)

    # whiten = (tensor - mean) * torch.rsqrt(var + eps)
    whiten = (tensor - mean) / (torch.sqrt(var) + eps)

    if not shift_mean:
        whiten = whiten + mean

    return whiten


def pad1d(tensor, pad, mode="constant", value=None):
    if mode != "constant" and tensor.ndim < 2:
        dtype = tensor.dtype
        tensor = tensor.to(torch.float32)
        return F.pad(tensor.unsqueeze(0), pad, mode=mode).squeeze(0).to(dtype)

    if mode != "constant":
        dtype = tensor.dtype
        tensor = tensor.to(torch.float32)
        return F.pad(tensor, pad, mode=mode).to(dtype)

    return F.pad(tensor, pad, mode, value)


def left_pad(tensor, length, mode="constant", value=None):
    return pad1d(tensor, (length - tensor.shape[1], 0), mode=mode, value=value)


def right_pad(tensor, length, mode="constant", value=None):
    return pad1d(tensor, (0, length - tensor.shape[1]), mode=mode, value=value)


def get_attention_mask(sample, attention_mask, pad_token_id):
    mask = (sample.not_equal(pad_token_id)).to(torch.int64)
    # mask = ((sample == eos_token_id).cumsum(-1) == 0).to(torch.int64)
    mask[:, : attention_mask.shape[1]] = attention_mask

    return mask


def truncate_penalty(samples, rewards, eoc_token_id, penalty):
    mask = (samples == eoc_token_id).sum(-1) == 0
    rewards = rewards.clone()
    rewards[mask] = penalty

    return penalty


def aggregate_loss(loss, mask, mode, max_tokens=None):
    if mode == "token-mean":
        return masked_mean(loss, mask)

    elif mode == "token-mean-seq-mean":
        loss = torch.sum(loss * mask, -1) / mask.sum(-1)

        return loss.mean()

    elif mode == "token-sum":
        batch_size = loss.shape[0]

        loss = torch.sum(loss * mask)

        if max_tokens is not None:
            loss = loss / (max_tokens * batch_size)

        else:
            loss = loss / batch_size

        return loss

    else:
        raise ValueError(f"Invalid mode: {mode}")
