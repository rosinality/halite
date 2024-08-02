import math

import torch
from torch.nn import init


def calculate_fan(
    tensor: torch.Tensor,
    mode: str,
    fan_in_dim: tuple[int, ...] = (1,),
    fan_out_dim: tuple[int, ...] = (0,),
    divide_dim: int = 1,
):
    mode = mode.lower()

    fan_in = 1
    for dim in fan_in_dim:
        fan_in *= tensor.shape[dim]

    fan_out = 1
    for dim in fan_out_dim:
        fan_out *= tensor.shape[dim]

    return fan_in // divide_dim if mode == "fan_in" else fan_out // divide_dim


@torch.no_grad()
def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    truncate: float | None = None,
    fan_in_dim: tuple[int, ...] = (1,),
    fan_out_dim: tuple[int, ...] = (0,),
    divide_dim: int = 1,
    generator: torch.Generator | None = None,
):
    fan = calculate_fan(tensor, mode, fan_in_dim, fan_out_dim, divide_dim)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    if truncate is not None:
        bound = truncate * std

        init.trunc_normal_(
            tensor, mean=0, std=std, a=-bound, b=bound, generator=generator
        )

    else:
        init.normal_(tensor, mean=0, std=std, generator=generator)
