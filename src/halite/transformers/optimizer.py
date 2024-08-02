import torch
from torch import distributed as dist
from torch.distributed._tensor import DTensor, Replicate


def calc_norm(x):
    return x.pow(2).sum(-2, keepdim=True).sqrt()

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
            # norm = calc_norm(p.data) # .norm(p=2, dim=-2, keepdim=True)
            # norm = p.data.norm(p=2, dim=-2, keepdim=True)
            norm = torch.linalg.vector_norm(p, ord=2, dim=0, keepdim=True)

        else:
            norm = torch.linalg.vector_norm(p, ord=2, dim=1, keepdim=True)
            # norm = p.data.pow(2).sum(-1, keepdim=True).sqrt()
            # norm = p.data.norm(p=2, dim=-1, keepdim=True)
            # norm = calc_norm(p.data.transpose(-2, -1).contiguous().transpose(-2, -1).contiguous()) # .norm(p=2, dim=-2, keepdim=True).transpose(-2, -1).contiguous()

        # rank = dist.get_rank()
        # if rank in (0, 1):
        #     print(dist.get_rank(), n, norm.sum())

        p.div_(norm)
