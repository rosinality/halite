from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
)

try:
    from torch.nn.attention.flex_attention import flex_attention

except ImportError:
    flex_attention = None

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input

except ImportError:
    pad_input = None

    pass

from halite.transformers.initialize import init_weights


class AttentionCache:
    def __init__(
        self, key: torch.Tensor, value: torch.Tensor, length: int, n_head: int
    ):
        self.key = key
        self.value = value
        self.length = length
        self.n_head = n_head

    def __getitem__(self, index):
        # key = self.key.reshape(-1, self.n_head, *self.key.shape[1:])
        # key = key.index_select(0, index).reshape(-1, *self.key.shape[1:])
        key = self.key.index_select(0, index)
        value = self.value.index_select(0, index)

        return AttentionCache(key, value, self.length, self.n_head)


class UnpadParams(NamedTuple):
    batch: int
    seqlen: int
    indices_q: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_length_q: int
    indices_k: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_length_k: int


def unpad_params(padding_mask):
    lengths = padding_mask.sum(-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_length = lengths.max().item()
    cu_seqlens = F.pad(torch.cumsum(lengths, 0, dtype=torch.int32), (1, 0))

    return UnpadParams(
        padding_mask.shape[0],
        padding_mask.shape[-1],
        indices,
        cu_seqlens,
        max_length,
        indices,
        cu_seqlens,
        max_length,
    )


def unpad_input(tensor, indices):
    if tensor.ndim < 3:
        tensor = tensor.unsqueeze(-1)

    return index_first_axis(
        tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:]), indices
    )


def build_unpad_params(padding_mask, query_len, key_len=None, batch_size=None):
    indices_k, cu_seqlens_k, max_length_k = unpad_params(padding_mask)

    if key_len is None or query_len == key_len:
        cu_seqlens_q = cu_seqlens_k
        max_length_q = max_length_k
        indices_q = indices_k

    elif query_len == 1:
        max_length_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=padding_mask.device
        )
        indices_q = cu_seqlens_q[:-1]

    else:
        padding_mask = padding_mask[:, -query_len:]
        indices_q, cu_seqlens_q, max_length_q = unpad_params(padding_mask)

    return UnpadParams(
        indices_q, cu_seqlens_q, max_length_q, indices_k, cu_seqlens_k, max_length_k
    )


def unpad_qkv(query, key, value, unpad_params):
    (
        indices_q,
        cu_seqlens_q,
        max_length_q,
        indices_k,
        cu_seqlens_k,
        max_length_k,
    ) = unpad_params
    query_len = query.shape[1]
    batch, key_len, n_heads, head_dim = key.shape

    key = index_first_axis(key.reshape(batch * key_len, n_heads, head_dim), indices_k)
    value = index_first_axis(
        value.reshape(batch * key_len, n_heads, head_dim), indices_k
    )

    if query_len == key_len or query_len != 1:
        query = index_first_axis(
            query.reshape(batch * key_len, n_heads, head_dim), indices_q
        )

    else:
        query = query.squeeze(1)

    return (
        query,
        key,
        value,
        indices_q,
        cu_seqlens_q,
        cu_seqlens_k,
        max_length_q,
        max_length_k,
    )


class SelfAttention(nn.Module):
    def __init__(
        self,
        qkv,
        attention,
        out,
        qkv_split="llama",
        qkv_init=None,
        out_init=None,
        layer_type=None,
    ):
        super().__init__()

        self.qkv = qkv
        self.attention = attention
        self.out = out

        self.n_heads = attention.n_heads
        self.head_dim = attention.head_dim

        self.n_key_value_heads = attention.n_heads

        if attention.n_key_value_heads is not None:
            self.n_key_value_heads = attention.n_key_value_heads

        self.qkv_out_dim = self.head_dim * (self.n_heads + self.n_key_value_heads * 2)

        self.qkv_split = {"megatron": self.qkv_megatron, "llama": self.qkv_llama}.get(
            qkv_split
        )
        self.qkv_init = qkv_init
        self.out_init = out_init
        self.layer_type = layer_type

    def init_weights(self):
        init_weights(self.qkv, self.qkv_init)
        init_weights(self.out, self.out_init)

    def qkv_megatron(self, qkv):
        qkv = qkv.reshape(
            qkv.shape[0],
            qkv.shape[1],
            self.n_key_value_heads,
            (self.n_heads // self.n_key_value_heads + 2) * self.head_dim,
        )
        q, k, v = qkv.split(
            (
                self.n_heads // self.n_key_value_heads * self.head_dim,
                self.head_dim,
                self.head_dim,
            ),
            -1,
        )

        return q, k, v

    def qkv_llama(self, qkv):
        tp = self.qkv_out_dim // qkv.shape[-1]

        q, k, v = qkv.split(
            (
                self.n_heads * self.head_dim // tp,
                self.n_key_value_heads * self.head_dim // tp,
                self.n_key_value_heads * self.head_dim // tp,
            ),
            -1,
        )

        return q, k, v

    def forward(
        self,
        input,
        attention_mask=None,
        attention_bias=None,
        pos_emb=None,
        cache=None,
        use_cache=False,
        unpad_params=None,
    ):
        qkv = self.qkv(input)

        q, k, v = self.qkv_split(qkv)

        if self.layer_type is not None:
            if isinstance(attention_mask, Mapping):
                attention_mask = attention_mask[self.layer_type]

            if isinstance(attention_bias, Mapping):
                attention_bias = attention_bias[self.layer_type]

            if isinstance(pos_emb, Mapping):
                pos_emb = pos_emb[self.layer_type]

        out, next_cache = self.attention(
            q,
            k,
            v,
            attention_mask,
            attention_bias,
            pos_emb,
            cache,
            use_cache,
            unpad_params=unpad_params,
        )

        out = self.out(out)

        return out, next_cache

    def parallelize_plan(self, **kwargs):
        return {
            ".": PrepareModuleInput(
                input_layouts=(Shard(1), None, None, None, None, None, None),
                desired_input_layouts=(Replicate(), None, None, None, None, None, None),
            ),
            "qkv": ColwiseParallel(),
            "out": RowwiseParallel(
                output_layouts=Shard(1),
            ),
        }


class SelfAttentionQKV(nn.Module):
    def __init__(
        self,
        q,
        k,
        v,
        attention,
        out,
        q_init=None,
        k_init=None,
        v_init=None,
        out_init=None,
        scaler=None,
        q_norm=None,
        k_norm=None,
        layer_type=None,
    ):
        super().__init__()

        self.q = q
        self.k = k
        self.v = v
        self.attention = attention
        self.out = out
        self.scaler = scaler

        self.n_heads = attention.n_heads
        self.head_dim = attention.head_dim
        self.n_key_value_heads = self.n_heads
        self.qkv_out_dim = self.head_dim * (self.n_heads + self.n_key_value_heads * 2)

        if attention.n_key_value_heads is not None:
            self.n_key_value_heads = attention.n_key_value_heads

        self.q_init = q_init
        self.k_init = k_init
        self.v_init = v_init
        self.out_init = out_init

        self.q_norm = q_norm
        self.k_norm = k_norm
        self.layer_type = layer_type

    def init_weights(self):
        init_weights(self.q, self.q_init)
        init_weights(self.k, self.k_init)
        init_weights(self.v, self.v_init)
        init_weights(self.out, self.out_init)

    def forward(
        self,
        input,
        attention_mask=None,
        attention_bias=None,
        pos_emb=None,
        cache=None,
        use_cache=False,
        unpad_params=None,
    ):
        q, k, v = self.q(input), self.k(input), self.v(input)

        if self.q_norm is not None:
            q = self.q_norm(q)

        if self.k_norm is not None:
            k = self.k_norm(k)

        if self.scaler is not None:
            q = self.scaler(q)
            k = self.scaler(k)

        if self.layer_type is not None:
            if isinstance(attention_mask, Mapping):
                attention_mask = attention_mask[self.layer_type]

            if isinstance(attention_bias, Mapping):
                attention_bias = attention_bias[self.layer_type]

            if isinstance(pos_emb, Mapping):
                pos_emb = pos_emb[self.layer_type]

        out, next_cache = self.attention(
            q,
            k,
            v,
            attention_mask,
            attention_bias,
            pos_emb,
            cache,
            use_cache,
            unpad_params=unpad_params,
        )

        out = self.out(out)

        return out, next_cache

    def parallelize_plan(self, **kwargs):
        return {
            ".": PrepareModuleInput(
                input_layouts=(Shard(1), None, None, None, None, None, None),
                desired_input_layouts=(Replicate(), None, None, None, None, None, None),
            ),
            "q": ColwiseParallel(),
            "k": ColwiseParallel(),
            "v": ColwiseParallel(),
            "out": RowwiseParallel(
                output_layouts=Shard(1),
            ),
        }


class Attention(nn.Module):
    def __init__(
        self,
        n_heads,
        head_dim,
        n_key_value_heads=None,
        attn_drop=0,
        is_causal=False,
        apply_pos_emb_fn=None,
        processor="auto",
        normalize=True,
        softcap=0.0,
        q_norm=None,
        k_norm=None,
        **processor_kwargs,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim

        self.attention_kwargs = processor_kwargs

        self.avail_torch = processor == "torch" or (
            hasattr(F, "scaled_dot_product_attention") and processor == "auto"
        )

        self.use_flash_attn = processor == "flash_attn"
        self.use_flex_attention = processor == "flex"
        if self.use_flex_attention:
            assert flex_attention is not None, "flex attention is not available"

        self.n_key_value_heads = n_key_value_heads
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.is_causal = is_causal
        self.apply_pos_emb_fn = apply_pos_emb_fn
        self.softcap = softcap

        self.q_norm = q_norm
        self.k_norm = k_norm

        self.normalize = normalize
        if isinstance(normalize, bool) and normalize:
            self.normalize = 1 / (head_dim**0.5)

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attention_bias=None,
        pos_embed=None,
        cache=None,
        use_cache=False,
        unpad_params=None,
    ):
        batch, query_length, _ = query.shape
        _, key_length, _ = key.shape

        query = query.view(batch, query_length, -1, self.head_dim)
        key = key.view(batch, key_length, -1, self.head_dim)
        value = value.view(batch, key_length, -1, self.head_dim)

        if self.q_norm is not None:
            query = self.q_norm(query)

        if self.k_norm is not None:
            key = self.k_norm(key)

        if self.apply_pos_emb_fn is not None:
            query, key = self.apply_pos_emb_fn(query, key, pos_embed)

        out, next_cache = self.attention(
            query,
            key,
            value,
            attention_mask,
            attention_bias,
            cache=cache,
            use_cache=use_cache,
            unpad_params=unpad_params,
        )

        if self.use_flash_attn:
            out = out.reshape(batch, query_length, -1)

        else:
            # batch, n_head, query_length, dim -> batch, query_length, n_head, dim
            out = out.permute(0, 2, 1, 3).contiguous()
            # batch, query_length, n_head, dim -> batch, query_length, n_head * dim
            out = out.reshape(batch, query_length, -1)

        return out, next_cache

    def attention(
        self,
        query,
        key,
        value,
        attention_mask,
        attention_bias,
        cache,
        use_cache,
        unpad_params=None,
    ):
        if self.use_flash_attn:
            return self.attention_flash(
                query,
                key,
                value,
                cache,
                use_cache,
                unpad_params=unpad_params,
            )

        if self.avail_torch:
            return self.attention_torch(
                query, key, value, attention_mask, attention_bias, cache, use_cache
            )

        if self.use_flex_attention:
            return self.attention_flex(
                query, key, value, attention_mask, cache, use_cache
            )

        return self.attention_native(
            query, key, value, attention_mask, attention_bias, cache, use_cache
        )

    @torch.compiler.disable
    def attention_flash_varlen(self, query, key, value, unpad_params):
        out = flash_attn_varlen_func(
            query,
            key,
            value,
            unpad_params.cu_seqlens_q,
            unpad_params.cu_seqlens_k,
            max_seqlen_q=unpad_params.max_length_q,
            max_seqlen_k=unpad_params.max_length_k,
            dropout_p=self.attn_drop_p if self.training else 0,
            causal=self.is_causal,
            softmax_scale=self.normalize,
            softcap=self.softcap,
            **self.attention_kwargs,
        )

        return out

    def attention_flash(
        self,
        query,
        key,
        value,
        cache,
        use_cache,
        unpad_params: UnpadParams | None = None,
    ):
        if cache is not None:
            key, value = cache.get(key, value)

        next_cache = None

        if use_cache:
            next_cache = (key, value)

        if unpad_params is not None:
            out = self.attention_flash_varlen(
                query.squeeze(0), key.squeeze(0), value.squeeze(0), unpad_params
            )

        else:
            out = flash_attn_func(
                query,
                key,
                value,
                self.attn_drop_p if self.training else 0,
                self.normalize,
                causal=self.is_causal,
                softcap=self.softcap,
                **self.attention_kwargs,
            )

        return out, next_cache

    def attention_torch(
        self, query, key, value, attention_mask, attention_bias, cache, use_cache
    ):
        # query: batch, query_length, n_head, dim
        # key: batch, key_length, n_head, dim
        # value: batch, key_length, n_head, dim
        # attention_mask: batch, n_head, query_length, key_length

        query = query.permute(0, 2, 1, 3)  # batch, n_head, query_length, dim
        key = key.permute(0, 2, 1, 3)  # batch, n_head, key_length, dim
        value = value.permute(0, 2, 1, 3)  # batch, n_head, key_length, dim

        if cache is not None:
            key = torch.cat((cache[0], key), 2)
            value = torch.cat((cache[1], value), 2)

        next_cache = None

        if use_cache:
            next_cache = (key, value)

        is_causal = self.is_causal

        if attention_bias is not None:
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_bias.masked_fill(
                    attention_mask, torch.finfo(attention_bias.dtype).min
                )

            else:
                attention_mask = attention_mask + attention_bias

            is_causal = False

        else:
            attention_mask = None if is_causal else attention_mask

        dropout = self.attn_drop_p if self.training else 0

        # batch, n_head, query_length, dim
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask,
            dropout,
            is_causal=is_causal,
            scale=self.normalize,
            enable_gqa=self.n_key_value_heads is not None,
        )

        return out, next_cache

    def attention_flex(self, query, key, value, attention_mask, cache, use_cache):
        # query: batch, query_length, n_head, dim
        # key: batch, key_length, n_head, dim
        # value: batch, key_length, n_head, dim
        # attention_mask: batch, n_head, query_length, key_length

        query = query.permute(0, 2, 1, 3)  # batch, n_head, query_length, dim
        key = key.permute(0, 2, 1, 3)  # batch, n_head, key_length, dim
        value = value.permute(0, 2, 1, 3)  # batch, n_head, key_length, dim

        if cache is not None:
            key = torch.cat((cache[0], key), 2)
            value = torch.cat((cache[1], value), 2)

        next_cache = None

        if use_cache:
            next_cache = (key, value)

        score_mod, block_mask = attention_mask

        # batch, n_head, query_length, dim
        out = flex_attention(
            query,
            key,
            value,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=self.normalize,
            enable_gqa=self.n_key_value_heads is not None,
        )

        return out, next_cache

    def attention_native(
        self, query, key, value, attention_mask, attention_bias, cache, use_cache
    ):
        # query: batch, query_length, n_head, dim
        # key: batch, key_length, n_head, dim
        # value: batch, key_length, n_head, dim
        # attention_mask: batch, n_head, query_length, key_length

        batch, query_length, n_head, dim = query.shape
        key_length = key.shape[1]

        if self.n_key_value_heads is not None:
            query = query.permute(0, 2, 1, 3).reshape(batch, n_head * query_length, dim)
            key = key.permute(0, 3, 1, 2).reshape(batch, dim, key_length)

        else:
            query = query.permute(0, 2, 1, 3).reshape(batch * n_head, query_length, dim)
            # key = key.permute(1, 2, 3, 0).reshape(batch * n_head, dim, key_length)
            # key = key.permute(1, 2, 0, 3)
            key = key.permute(0, 2, 1, 3)

        # batch, n_head, key_length, dim
        value = value.permute(0, 2, 1, 3)

        if cache is not None:
            key = torch.cat((cache[0], key), 2)
            value = torch.cat((cache[1], value), 2)

        next_cache = None

        if use_cache:
            next_cache = (key, value)

        # this .contiguous() makes difference. why?
        key = key.transpose(2, 3).reshape(batch * n_head, dim, -1).contiguous()
        # batch, n_head, key_length, dims
        key_length = key.shape[2]

        # batch * n_head, query_length, key_length

        if attention_bias is None:
            input = query.new_empty(query.shape[0], query.shape[1], key.shape[2])
            beta = 0

        else:
            input = attention_bias.expand(
                batch, n_head, query_length, key_length
            ).reshape(batch * n_head, query_length, key_length)
            beta = 1

        attn_score = torch.baddbmm(input, query, key, beta=beta, alpha=self.normalize)
        attn_score = attn_score.view(batch, n_head, query_length, key_length)

        if attn_score.dtype == torch.float16:
            attn_score = attn_score.to(torch.float32)

        if attention_mask is not None:
            attn_score = attn_score.masked_fill(
                attention_mask, torch.finfo(attn_score.dtype).min
            )

        attn = torch.softmax(attn_score, -1, dtype=torch.float32).to(input.dtype)
        attn = self.attn_drop(attn)

        # (batch, n_head, query_length, key_length) @ (batch, n_head, key_length, dim)
        out = attn @ value

        return out, next_cache


class DiffAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        head_dim,
        lambda_init,
        sub_norm,
        n_key_value_heads=0,
        attn_drop=0,
        is_causal=False,
        apply_pos_emb_fn=None,
        processor="auto",
        normalize=True,
        softcap=0.0,
        **processor_kwargs,
    ):
        super().__init__()

        self.n_heads = n_heads * 2
        self.head_dim = head_dim

        self.attention_kwargs = processor_kwargs

        self.avail_torch = processor == "torch" or (
            hasattr(F, "scaled_dot_product_attention") and processor == "auto"
        )

        self.use_flash_attn = processor == "flash_attn"

        self.n_key_value_heads = n_key_value_heads
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.is_causal = is_causal
        self.apply_pos_emb_fn = apply_pos_emb_fn
        self.softcap = softcap

        self.normalize = normalize
        if isinstance(normalize, bool) and normalize:
            self.normalize = 1 / (head_dim**0.5)

        self.register_buffer(
            "lambda_init",
            torch.as_tensor(lambda_init, dtype=torch.float32),
            persistent=True,
        )
        self.lambda_q1 = nn.Parameter(torch.empty(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.empty(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.empty(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.empty(self.head_dim))

        self.sub_norm = sub_norm

    def init_weights(self):
        nn.init.normal_(self.lambda_q1, mean=0, std=0.1)
        nn.init.normal_(self.lambda_k1, mean=0, std=0.1)
        nn.init.normal_(self.lambda_q2, mean=0, std=0.1)
        nn.init.normal_(self.lambda_k2, mean=0, std=0.1)

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attention_bias=None,
        pos_embed=None,
        cache=None,
        use_cache=False,
        unpad_params=None,
    ):
        batch, query_length, _ = query.shape
        _, key_length, _ = key.shape

        query = query.view(batch, query_length, -1, self.head_dim)
        key = key.view(batch, key_length, -1, self.head_dim)
        value = value.view(batch, key_length, -1, self.head_dim * 2)

        if self.apply_pos_emb_fn is not None:
            query, key = self.apply_pos_emb_fn(query, key, pos_embed)

        attn1, attn2, next_cache = self.attention(
            query,
            key,
            value,
            attention_mask,
            attention_bias,
            cache=cache,
            use_cache=use_cache,
            unpad_params=unpad_params,
        )

        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).to(torch.float32)
        ).type_as(query)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).to(torch.float32)
        ).type_as(query)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn = attn1 - lambda_full * attn2
        attn = self.sub_norm(attn)
        out = (1 - self.lambda_init) * attn

        if self.use_flash_attn:
            out = out.reshape(batch, query_length, -1)

        else:
            # batch, n_head, query_length, dim -> batch, query_length, n_head, dim
            out = out.permute(0, 2, 1, 3).contiguous()
            # batch, query_length, n_head, dim -> batch, query_length, n_head * dim
            out = out.reshape(batch, query_length, -1)

        return out, next_cache

    def attention(
        self,
        query,
        key,
        value,
        attention_mask,
        attention_bias,
        cache,
        use_cache,
        unpad_params=None,
    ):
        if self.use_flash_attn:
            return self.attention_flash(
                query, key, value, cache, use_cache, unpad_params=unpad_params
            )

        if self.avail_torch:
            return self.attention_torch(
                query, key, value, attention_mask, attention_bias, cache, use_cache
            )

        return self.attention_native(
            query, key, value, attention_mask, attention_bias, cache, use_cache
        )

    def attention_flash(
        self,
        query,
        key,
        value,
        cache,
        use_cache,
        unpad_params: UnpadParams | None = None,
    ):
        if cache is not None:
            key, value = cache.get(key, value)

        next_cache = None

        if use_cache:
            next_cache = (key, value)

        if unpad_params is not None:
            out = flash_attn_varlen_func(
                query.squeeze(0),
                key.squeeze(0),
                value.squeeze(0),
                unpad_params.cu_seqlens_q,
                unpad_params.cu_seqlens_k,
                max_seqlen_q=unpad_params.max_length_q,
                max_seqlen_k=unpad_params.max_length_k,
                dropout_p=self.attn_drop_p if self.training else 0,
                causal=self.is_causal,
                softmax_scale=self.normalize,
                softcap=self.softcap,
                **self.attention_kwargs,
            )

        else:
            out = flash_attn_func(
                query,
                key,
                value,
                self.attn_drop_p if self.training else 0,
                self.normalize,
                causal=self.is_causal,
                softcap=self.softcap,
                **self.attention_kwargs,
            )

        return out, next_cache

    def attention_torch(
        self, query, key, value, attention_mask, attention_bias, cache, use_cache
    ):
        # query: batch, query_length, n_head, dim
        # key: batch, key_length, n_head, dim
        # value: batch, key_length, n_head, dim
        # attention_mask: batch, n_head, query_length, key_length

        if cache is not None:
            key = torch.cat((cache[0], key), 1)
            value = torch.cat((cache[1], value), 1)

        next_cache = None

        if use_cache:
            next_cache = (key, value)

        batch, query_length = query.shape[:2]
        key_length = key.shape[1]

        query = query.view(
            batch, query_length, self.n_heads // 2, 2, self.head_dim
        ).permute(0, 2, 3, 1, 4)  # batch, n_head, 2, query_length, dim
        key = key.view(batch, key_length, self.n_heads // 2, 2, self.head_dim).permute(
            0, 2, 3, 1, 4
        )  # batch, n_head, 2, key_length, dim
        value = value.permute(0, 2, 1, 3)  # batch, n_head, key_length, dim

        q1, q2 = query.unbind(2)
        k1, k2 = key.unbind(2)

        is_causal = self.is_causal

        if attention_bias is not None:
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_bias.masked_fill(
                    attention_mask, torch.finfo(attention_bias.dtype).min
                )

            else:
                attention_mask = attention_mask + attention_bias

            is_causal = False

        else:
            attention_mask = None if is_causal else attention_mask

        dropout = self.attn_drop_p if self.training else 0

        # batch, n_head, query_length, dim
        attn1 = F.scaled_dot_product_attention(
            q1, k1, value, attention_mask, dropout, is_causal=is_causal
        )
        attn2 = F.scaled_dot_product_attention(
            q2, k2, value, attention_mask, dropout, is_causal=is_causal
        )

        return attn1, attn2, next_cache

    def attention_native(
        self, query, key, value, attention_mask, attention_bias, cache, use_cache
    ):
        # query: batch, query_length, n_head, dim
        # key: batch, key_length, n_head, dim
        # value: batch, key_length, n_head, dim
        # attention_mask: batch, n_head, query_length, key_length

        batch, query_length, n_head, dim = query.shape
        key_length = key.shape[1]

        if self.n_key_value_heads > 0:
            query = query.permute(0, 2, 1, 3).reshape(batch, n_head * query_length, dim)
            key = key.permute(0, 3, 1, 2).reshape(batch, dim, key_length)

        else:
            query = query.permute(0, 2, 1, 3).reshape(batch * n_head, query_length, dim)
            # key = key.permute(1, 2, 3, 0).reshape(batch * n_head, dim, key_length)
            # key = key.permute(1, 2, 0, 3)
            key = key.permute(0, 2, 1, 3)

        # batch, n_head, key_length, dim
        value = value.permute(0, 2, 1, 3)

        if cache is not None:
            key = torch.cat((cache[0], key), 2)
            value = torch.cat((cache[1], value), 2)

        next_cache = None

        if use_cache:
            next_cache = (key, value)

        # this .contiguous() makes difference. why?
        key = key.transpose(2, 3).reshape(batch * n_head, dim, -1).contiguous()
        # batch, n_head, key_length, dims
        key_length = key.shape[2]

        # batch * n_head, query_length, key_length

        if attention_bias is None:
            input = query.new_empty(query.shape[0], query.shape[1], key.shape[2])
            beta = 0

        else:
            input = attention_bias.expand(
                batch, n_head, query_length, key_length
            ).reshape(batch * n_head, query_length, key_length)
            beta = 1

        attn_score = torch.baddbmm(input, query, key, beta=beta, alpha=self.normalize)
        attn_score = attn_score.view(batch, n_head, query_length, key_length)

        if attn_score.dtype == torch.float16:
            attn_score = attn_score.to(torch.float32)

        if attention_mask is not None:
            attn_score = attn_score.masked_fill(
                attention_mask, torch.finfo(attn_score.dtype).min
            )

        attn = torch.softmax(attn_score, -1, dtype=torch.float32).to(input.dtype)
        attn = self.attn_drop(attn)

        # (batch, n_head, query_length, key_length) @ (batch, n_head, key_length, dim)
        out = attn @ value

        return out, next_cache
