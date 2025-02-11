from torch.distributed.tensor import distribute_tensor, DTensor


def reshard_state_dict(source_state_dict: dict, target_state_dict: dict):
    resharded = {}

    for k, target_tensor in target_state_dict.items():
        tensor = source_state_dict[k]

        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()

        if isinstance(target_tensor, DTensor):
            tensor = distribute_tensor(
                tensor, target_tensor.device_mesh, target_tensor.placements
            )

        resharded[k] = tensor

    return resharded
