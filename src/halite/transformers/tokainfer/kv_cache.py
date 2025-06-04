import torch
from torch import Tensor, nn

from halite.transformers.tokainfer.types import DeviceType


class LayerKVCache(nn.Module):
    k_cache: Tensor
    v_cache: Tensor | None

    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
        num_pages: int,
        page_size: int,
        device: DeviceType | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_pages = num_pages
        self.page_size = page_size

        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim

        self.register_buffer(
            "k_cache",
            torch.zeros(
                (
                    num_pages,
                    page_size,
                    num_kv_heads,
                    head_dim,
                ),
                device=device,
                dtype=dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(
                (
                    num_pages,
                    page_size,
                    num_kv_heads,
                    head_dim,
                ),
                device=device,
                dtype=dtype,
            ),
            persistent=False,
        )
