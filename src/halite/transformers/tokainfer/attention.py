from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from halite.transformers.attention import SelfAttention, SelfAttentionQKV
from halite.transformers.tokainfer.attention_fn import (
    tokainfer_attention,
)
from halite.transformers.tokainfer.kv_cache import LayerKVCache
from halite.transformers.tokainfer.types import (
    AttentionInfo,
    BatchState,
    WrapperCollection,
)


class TokaInferSelfAttention(SelfAttention):
    def forward(self, input, batch, pos_emb=None):
        qkv = self.qkv(input)
        q, k, v = self.qkv_split(qkv)
        out = self.attention(q, k, v, batch, pos_emb)
        out = self.out(out)

        return out


class TokaInferSelfAttentionQKV(SelfAttentionQKV):
    def forward(self, input, batch, pos_emb=None):
        q, k, v = self.q(input), self.k(input), self.v(input)
        out = self.attention(q, k, v, batch, pos_emb)
        out = self.out(out)

        return out


class TokaInferAttention(nn.Module):
    def __init__(
        self,
        layer_id: int,
        n_heads: int,
        head_dim: int,
        n_key_value_heads: int | None = None,
        apply_pos_emb_fn: Callable = None,
        q_norm=None,
        k_norm=None,
    ):
        super().__init__()

        self.layer_id = layer_id

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_key_value_heads = n_heads

        if n_key_value_heads is not None:
            self.n_key_value_heads = n_key_value_heads

        self.apply_pos_emb_fn = apply_pos_emb_fn

        self.q_norm = q_norm
        self.k_norm = k_norm

        self.layer_cache: LayerKVCache | None = None
        self.wrapper_collection: WrapperCollection | None = None
        self.attention_info: AttentionInfo | None = None

        self.attn_fn = self.build_attn_fn()

    def build_attn_fn(self):
        @torch.library.custom_op(
            f"halite::tokainfer_attention_layer_{self.layer_id}",
            mutates_args=("k_cache", "v_cache"),
        )
        def attn_fn(
            ragged_q: torch.Tensor,
            ragged_k: torch.Tensor,
            ragged_v: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
        ) -> torch.Tensor:
            num_padding = self.attention_info.num_padding

            if num_padding > 0:
                ragged_q = ragged_q[:-num_padding]
                ragged_k = ragged_k[:-num_padding]
                ragged_v = ragged_v[:-num_padding]

            out = tokainfer_attention(
                ragged_q=ragged_q,
                ragged_k=ragged_k,
                ragged_v=ragged_v,
                k_cache=k_cache,
                v_cache=v_cache,
                attn_info=self.attention_info,
                wrappers=self.wrapper_collection,
            )

            if num_padding > 0:
                out = F.pad(out, (0, 0, 0, 0, 0, num_padding))

            return out

        @attn_fn.register_fake
        def _(
            ragged_q: torch.Tensor,
            ragged_k: torch.Tensor,
            ragged_v: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
        ) -> torch.Tensor:
            return torch.empty_like(ragged_q)

        return attn_fn

    def forward(self, query, key, value, batch_state: BatchState, pos_emb=None):
        batch_size = query.shape[0]

        query = query.view(-1, self.n_heads, self.head_dim)
        key = key.view(-1, self.n_key_value_heads, self.head_dim)
        value = value.view(-1, self.n_key_value_heads, self.head_dim)

        if self.q_norm is not None:
            query = self.q_norm(query)

        if self.k_norm is not None:
            key = self.k_norm(key)

        if self.apply_pos_emb_fn is not None:
            query, key = self.apply_pos_emb_fn(query, key, pos_emb)

        raw_attn_output = self.attn_fn(
            ragged_q=query,
            ragged_k=key,
            ragged_v=value,
            k_cache=self.layer_cache.k_cache,
            v_cache=self.layer_cache.v_cache,
        ).clone()

        attn_output = raw_attn_output.view(batch_size, -1)

        return attn_output
