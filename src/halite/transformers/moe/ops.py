import torch


# bincount is not compilable
@torch.library.custom_op("halite::bincount", mutates_args={})
def bincount(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return x.bincount(minlength=minlength)


@bincount.register_fake
def _(x: torch.Tensor, minlength: int) -> torch.Tensor:
    return torch.empty(minlength, dtype=torch.long, device=x.device)
