from typing import Callable

import torch
from torch import nn

from halite.transformers.attention import SelfAttention, SelfAttentionQKV
from halite.transformers.infer.engine.batch import Batch, ForwardMode


class InferSelfAttention(SelfAttention):
    def forward(self, input, batch, pos_emb=None):
        qkv = self.qkv(input)
        q, k, v = self.qkv_split(qkv)
        out = self.attention(q, k, v, batch, pos_emb)
        out = self.out(out)

        return out


class InferSelfAttentionQKV(SelfAttentionQKV):
    def forward(self, input, batch, pos_emb=None):
        q, k, v = self.q(input), self.k(input), self.v(input)
        out = self.attention(q, k, v, batch, pos_emb)
        out = self.out(out)

        return out


class FlashInferAttention(nn.Module):
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

        self.attention_backend = None
        self.kv_pool = None
        self.attention_fn = self.build_attention_fn()

    def _build_attention_fn(self):
        @torch.library.custom_op(
            f"halite::flash_infer_attention_{self.layer_id}",
            mutates_args=("k_cache", "v_cache"),
        )
        def prefill_fn(
            query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor
        ) -> torch.Tensor:
            prefill, _ = self.attention_backend.get_wrappers()

            o = prefill.run(query.contiguous(), (k_cache, v_cache))

            return o

        @prefill_fn.register_fake
        def _(
            query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor
        ) -> torch.Tensor:
            return torch.empty_like(query)

        @torch.library.custom_op(
            f"halite::flash_infer_attention_decode_{self.layer_id}",
            mutates_args=("k_cache", "v_cache"),
        )
        def decode_fn(
            query: torch.Tensor,
            k_cache: torch.Tensor,
            v_cache: torch.Tensor,
        ) -> torch.Tensor:
            _, decode = self.attention_backend.get_wrappers()

            o = decode.run(query.contiguous(), (k_cache, v_cache))

            return o

        @decode_fn.register_fake
        def _(
            query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor
        ) -> torch.Tensor:
            return torch.empty_like(query)

        return prefill_fn, decode_fn

    def build_attention_fn(self):
        @torch.library.custom_op(
            f"halite::flash_infer_attention_{self.layer_id}",
            mutates_args=(),
        )
        def attention_fn(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_pool_ids: torch.Tensor,
            is_decode: bool,
        ) -> torch.Tensor:
            if key is not None:
                self.kv_pool.set_kv_buffer(self.layer_id, kv_pool_ids, key, value)

            prefill, decode = self.attention_backend.get_wrappers()

            if not is_decode:
                o = prefill.run(
                    query.contiguous().view(
                        -1,
                        self.attention_backend.n_qo_heads,
                        self.attention_backend.head_dim,
                    ),
                    self.kv_pool.get_kv_buffer(self.layer_id),
                )

            else:
                _, decode = self.attention_backend.get_wrappers()

                o = decode.run(
                    query.contiguous().view(
                        -1,
                        self.attention_backend.n_qo_heads,
                        self.attention_backend.head_dim,
                    ),
                    self.kv_pool.get_kv_buffer(self.layer_id),
                )

            return o

        @attention_fn.register_fake
        def _(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_pool_ids: torch.Tensor,
            is_decode: bool,
        ) -> torch.Tensor:
            return torch.empty_like(query)

        return attention_fn

    def forward(self, query, key, value, batch: Batch, pos_emb=None):
        query = query.view(-1, self.n_heads, self.head_dim)
        key = key.view(-1, self.n_key_value_heads, self.head_dim)
        value = value.view(-1, self.n_key_value_heads, self.head_dim)

        if self.q_norm is not None:
            query = self.q_norm(query)

        if self.k_norm is not None:
            key = self.k_norm(key)

        if self.apply_pos_emb_fn is not None:
            query, key = self.apply_pos_emb_fn(query, key, pos_emb)

        return self.attention_fn(
            query, key, value, batch.kv_pool_ids, batch.mode == ForwardMode.DECODE
        ).view(-1, self.n_heads * self.head_dim)

        # if batch.mode == ForwardMode.EXTEND:
        #     return self.forward_extend(query, key, value, batch)

        # elif batch.mode == ForwardMode.DECODE:
        #     return self.forward_decode(query, key, value, batch)

    def forward_extend(self, query, key, value, batch):
        # prefill, _ = batch.attention_backend.get_wrapper()
        kv_pool_ids = batch.kv_pool_ids

        if key is not None:
            batch.kv_pool.set_kv_buffer(self.layer_id, kv_pool_ids, key, value)

        """o = prefill.run(
            query.contiguous().view(
                -1, batch.attention_backend.n_qo_heads, batch.attention_backend.head_dim
            ),
            batch.kv_pool.get_kv_buffer(self.layer_id),
        )"""

        o = self.prefill_fn(
            query.contiguous().view(
                -1, batch.attention_backend.n_qo_heads, batch.attention_backend.head_dim
            ),
            *batch.kv_pool.get_kv_buffer(self.layer_id),
        )

        return o.view(
            -1,
            batch.attention_backend.n_qo_heads * batch.attention_backend.head_dim,
        )

    def forward_decode(self, query, key, value, batch):
        # _, decode = batch.attention_backend.get_wrapper()

        if key is not None:
            batch.kv_pool.set_kv_buffer(self.layer_id, batch.kv_pool_ids, key, value)

        """o = decode.run(
            query.contiguous().view(
                -1, batch.attention_backend.n_qo_heads, batch.attention_backend.head_dim
            ),
            batch.kv_pool.get_kv_buffer(self.layer_id),
        )"""

        o = self.decode_fn(
            query.contiguous().view(
                -1, batch.attention_backend.n_qo_heads, batch.attention_backend.head_dim
            ),
            *batch.kv_pool.get_kv_buffer(self.layer_id),
        )

        return o.view(
            -1, batch.attention_backend.n_qo_heads * batch.attention_backend.head_dim
        )
