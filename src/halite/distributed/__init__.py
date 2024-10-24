import torch
from torch import distributed as dist

def all_reduce_mean(tensor, group, world_size):
    dist.all_reduce(tensor, group=group)
    tensor = tensor / world_size

    return tensor


def all_reduce_flag(flag, group, device):
    flag_tensor = torch.tensor(int(flag), dtype=torch.float32, device=device)
    
    dist.all_reduce(flag_tensor, group=group, op=dist.ReduceOp.MAX)
    flag = flag_tensor.item() > 0

    return flag