import torch
from torch.distributed.tensor import distribute_tensor, DTensor


@torch.no_grad()
def reshard_state_dict(
    source_state_dict: dict, target_state_dict: dict, strip_fsdp_prefix: bool = True
):
    resharded = {}

    if strip_fsdp_prefix:
        source_state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in source_state_dict.items()
        }

    for k, target_tensor in target_state_dict.items():
        tensor = source_state_dict[k].to(target_tensor.dtype)

        if isinstance(tensor, DTensor):
            if isinstance(target_tensor, DTensor):
                tensor = tensor.redistribute(
                    target_tensor.device_mesh, target_tensor.placements
                )

                resharded[k] = tensor

                continue

            tensor = tensor.full_tensor()

        if isinstance(target_tensor, DTensor):
            tensor = distribute_tensor(
                tensor, target_tensor.device_mesh, target_tensor.placements
            )

        resharded[k] = tensor

    return resharded
