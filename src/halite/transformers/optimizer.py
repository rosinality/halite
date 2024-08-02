import torch
from torch import distributed as dist
from torch.distributed._tensor import DTensor, Replicate


def calc_norm(x):
    return x.pow(2).sum(-2, keepdim=True).sqrt()


def normalize(x, dim=-1):
    dtype = x.dtype
    x = x.float()
    res = (x / x.norm(p=2, dim=dim, keepdim=True)).to(dtype)

    return res


@torch.no_grad()
def ngpt_normalize(model, transposed=("out.weight",)):
    for n, p in model.named_parameters():
        if p.ndim < 2:
            continue

        is_transposed = False
        for t in transposed:
            if t in n:
                is_transposed = True

                break

        if is_transposed:
            p.copy_(normalize(p, 0))

        else:
            p.copy_(normalize(p, 1))
