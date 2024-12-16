import torch


class RequestToTokenPool:
    def __init__(self, max_size, max_context_len, device):
        self.max_size = max_size
        self.max_context_len = max_context_len
        self.device = device

        self.request_to_token = torch.zeros(
            max_size, max_context_len, dtype=torch.int32, device=device
        )
        self.free_slots = list(range(max_size))
        self.write_records = []

    def write(self, indices, values):
        self.request_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, size):
        select_index = self.free_slots[:size]
        self.free_slots = self.free_slots[size:]

        return select_index

    def free(self, indices):
        if isinstance(indices, int):
            self.free_slots.append(indices)

        else:
            self.free_slots.extend(indices)

    def clear(self):
        self.free_slots = list(range(self.max_size))
        self.write_records = []


class KVPool:
    pass


class MHAKVPool(KVPool):
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        n_heads: int,
        head_dim: int,
        n_layers: int,
        device: str,
    ):
        self.size = size
        self.device = device

        self.k_buffer = torch.empty(
            n_layers, size + 1, n_heads, head_dim, dtype=dtype, device=device
        ).unbind(0)
        self.v_buffer = torch.empty(
            n_layers, size + 1, n_heads, head_dim, dtype=dtype, device=device
        )

        self.is_in_free_group = False
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int32, device="cpu"
        )
        self.free_group = []

        self.clear()

    def alloc(self, size):
        if size > self.free_slots.shape[0]:
            return None

        index = self.free_slots[:size]
        self.free_slots = self.free_slots[size:]

        return index.to(self.device, non_blocking=True)

    def available_size(self):
        return len(self.free_slots)

    def clear(self):
        self.free_slots = torch.arange(1, self.size + 1, dtype=torch.int32)

    def get_kv_buffer(self, layer_id: int):
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def set_kv_buffer(self, layer_id, loc, k, v):
        self.k_buffer[layer_id][loc] = k
        self.v_buffer[layer_id][loc] = v

    def free(self, index: torch.Tensor):
        if self.is_in_free_group:
            self.free_group.append(index)

        else:
            self.free_slots = torch.cat((self.free_slots, index.cpu()))

    def free_group_begin(self):
        self.is_in_free_group = True
        self.free_group = []

    def free_group_end(self):
        self.is_in_free_group = False

        if self.free_group:
            self.free(torch.cat(self.free_group))
