import torch


class AttentionCache:
    def __init__(self, key: torch.Tensor, value: torch.Tensor, length_dim=2):
        self.key = key
        self.value = value
        self.length_dim = length_dim
        self.length = key.shape[self.length_dim]

    def __getitem__(self, index):
        key = self.key.index_select(0, index)
        value = self.value.index_select(0, index)

        return AttentionCache(key, value, self.length_dim)

    def get(self, key, value):
        return torch.cat((self.key, key), self.length_dim), torch.cat(
            (self.value, value), self.length_dim
        )


class BasicCacheManager:
    def __init__(self, n_caches, length_dim=2):
        self.length = 0
        self.caches = [None] * n_caches
        self.length_dim = length_dim

    def __getitem__(self, index):
        return self.caches[index]

    def get_cache(self, index, **kwargs):
        return self.__getitem__(index)

    def update(self, index, cache):
        self.caches[index] = AttentionCache(*cache, self.length_dim)
        self.length = cache[0].shape[self.length_dim]

    def select(self, index):
        self.caches = [cache[index] for cache in self.caches]


class AllocatedCache:
    def __init__(self, key: torch.Tensor, value: torch.Tensor, length: int):
        self.key = key
        self.value = value
        self.length = length

    def get(self, key, value):
        batch = key.shape[0]
        key_length = key.shape[2]

        self.key[
            :batch,
            :,
            self.length : self.length + key_length,
        ] = key
        self.value[
            :batch,
            :,
            self.length : self.length + key_length,
        ] = value

        key = self.key[:batch, :, : self.length + key_length]
        value = self.value[:batch, :, : self.length + key_length]

        self.length += key_length

        return key, value


class AllocatedCacheManager:
    def __init__(
        self,
        n_cache,
        max_batch_size,
        max_length,
        n_head,
        head_dim,
        dtype=torch.float32,
        device="cuda",
        caches=None,
    ):
        self.n_cache = n_cache
        self.lengths = [0] * self.n_cache
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.n_head = n_head
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.caches = caches

        if caches is None:
            self.caches = []
            for _ in range(self.n_cache):
                self.caches.append(
                    (
                        torch.empty(
                            max_batch_size,
                            n_head,
                            max_length,
                            head_dim,
                            dtype=dtype,
                            device=device,
                        ),
                        torch.empty(
                            max_batch_size,
                            n_head,
                            max_length,
                            head_dim,
                            dtype=dtype,
                            device=device,
                        ),
                    )
                )

    def reset(self):
        self.lengths = [0] * self.n_cache

    def get_view(self, batch_size, batch_start=0, start_length=0):
        caches = []

        for key, value in self.caches:
            caches.append(
                (
                    key[batch_start : batch_start + batch_size, :, start_length:],
                    value[batch_start : batch_start + batch_size, :, start_length:],
                )
            )

        cache_manager = AllocatedCacheManager(
            self.n_cache,
            self.max_batch_size,
            self.max_length,
            self.n_head,
            self.head_dim,
            self.dtype,
            self.device,
            caches=caches,
        )
        cache_manager.lengths = [start_length] * self.n_cache

        return cache_manager

    def select(self, index):
        end = index.shape[0]

        for key, value in self.caches:
            key[:end] = key[index]
            value[:end] = value[index]

    def to(self, *args, **kwargs):
        self.caches = [
            (key.to(*args, **kwargs), value.to(*args, **kwargs))
            for key, value in self.caches
        ]

    def __getitem__(self, index):
        key, value = self.caches[index]

        return AllocatedCache(key, value, self.lengths[index])

    def shift(self, size):
        length = self.lengths[0]

        for key, value in self.caches:
            key[:, :, : length - size] = key[:, :, size:length]
            value[:, :, : length - size] = key[:, :, size:length]

        self.lengths = [length - size for length in self.lengths]

    def set_cache(self, caches, current_position, batch_start=0):
        for (cache_k, cache_v), (key, value) in zip(self.caches, caches):
            batch, _, length, _ = key.shape
            cache_k[
                batch_start : batch_start + batch, current_position - length :
            ] = key
            cache_v[
                batch_start : batch_start + batch, current_position - length :
            ] = value

    def update(self, index, cache):
        self.lengths[index] = cache[0].shape[2]
