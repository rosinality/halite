from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

from halite.utils import get_torch_dtype


def apply_fsdp(model, dp_mesh, param_dtype, reduce_dtype):
    if param_dtype is not None:
        param_dtype = get_torch_dtype(param_dtype)

    if reduce_dtype is not None:
        reduce_dtype = get_torch_dtype(reduce_dtype)

    mixed_precision = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype
    )

    for i, block in model.blocks.items():
        reshard_after_forward = int(i) < len(model.blocks) - 1

        fully_shard(
            block,
            mesh=dp_mesh,
            mp_policy=mixed_precision,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(
        model, mesh=dp_mesh, mp_policy=mixed_precision, reshard_after_forward=True
    )

    return model
