import torch


def get_available_gpu_memory(device, gpu_id, distributed=False):
    if device == "cuda":
        n_gpus = torch.cuda.device_count()
        torch.cuda.empty_cache()
        free_memory, _ = torch.cuda.mem_get_info(gpu_id)

    if distributed:
        buffer = torch.tensor(free_memory, dtype=torch.float32).to(
            torch.device(device, gpu_id)
        )
        torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.MIN)
        free_memory = buffer.item()

    return free_memory / (1 << 30)
