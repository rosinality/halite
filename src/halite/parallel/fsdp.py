import torch
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.replicate import replicate

from halite.utils import get_torch_dtype


def apply_fsdp(model, dp_mesh, param_dtype, reduce_dtype, reshard_after_forward=True):
    if param_dtype is not None:
        param_dtype = get_torch_dtype(param_dtype)

    if reduce_dtype is not None:
        reduce_dtype = get_torch_dtype(reduce_dtype)

    mixed_precision = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype
    )

    for i, block in model.blocks.items():
        # reshard_after_forward = int(i) < len(model.blocks) - 1
        layer_reshard_after_forward = False
        if reshard_after_forward:
            layer_reshard_after_forward = True

        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mixed_precision,
            reshard_after_forward=layer_reshard_after_forward,
        )

    fully_shard(model, mesh=dp_mesh, mp_policy=mixed_precision)

    return model


def apply_ddp(model, dp_mesh, compile=False, compile_autograd=False):
    if compile:
        if compile_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )

        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
