import math
from operator import inv

import torch
from torch import nn


def make_causal_mask(batch_size, query_length, key_length, device):
    if query_length <= 1:
        return None

    mask = torch.ones((query_length, key_length), dtype=torch.bool, device=device)
    mask = torch.triu(mask, 1)

    return mask.expand(batch_size, 1, mask.shape[0], mask.shape[1])


def alibi_params(n_heads, tensor_parallel_rank, tensor_parallel_world_size):
    n_heads_per_tp = n_heads // tensor_parallel_world_size
    alibi_ratio = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
    alibi_start = alibi_ratio * alibi_ratio ** (n_heads_per_tp * tensor_parallel_rank)

    return alibi_start, alibi_ratio


def alibi_slopes(n_heads, device="cuda"):
    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        device=device,
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=device)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != n_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=device,
            dtype=torch.float32,
        )
        n_remains = min(closest_power_of_2, n_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1, 1 + 2 * n_remains, 2, dtype=torch.int32, device=device
        )
        slopes = torch.cat((slopes, torch.pow(extra_base, extra_powers)), 0)

    return slopes


class ALiBi(nn.Module):
    attention_bias: bool = True
    layer_shared: bool = True

    def __init__(self, n_heads, attention_mask=None, device="cuda"):
        super().__init__()

        slopes = alibi_slopes(n_heads, device=device)

        self.slopes = slopes[..., None]
        self.n_heads = n_heads
        self.pos_embed = None

        if attention_mask is not None:
            self.register_buffer(
                "pos_embed", self.generate_embedding(attention_mask), persistent=True
            )

    def generate_embedding(self, attention_mask):
        indexes = ((attention_mask.cumsum(-1) - 1) * attention_mask)[:, None, :]

        if self.slopes.device != indexes.device:
            self.slopes = self.slopes.to(device=indexes.device)

        alibi = self.slopes * indexes

        return alibi.reshape(
            attention_mask.shape[0], self.n_heads, 1, attention_mask.shape[1]
        )

    def forward(self, attention_mask):
        if self.pos_embed is not None:
            return self.pos_embed

        return self.generate_embedding(attention_mask)


class RotaryEmbedding(nn.Module):
    attention_bias: bool = False
    layer_shared: bool = True

    def __init__(
        self, dim, max_position_embeddings, base=10000, device=None, use_complex=False
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device
        self.use_complex = use_complex

        self.init_weights()

    def init_weights(self):
        inv_freq = self.get_inv_freq()
        self.inv_freq = inv_freq

        self.init_cache(self.max_position_embeddings, "cpu", torch.float32)

    def get_inv_freq(self):
        inv_freq = 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device="cpu")
                / self.dim
            )
        )

        return inv_freq

    def init_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)

        if self.use_complex:
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            self.freqs_cis = freqs_cis

            return

        cos, sin = freqs.cos(), freqs.sin()
        self.freqs_cis = torch.stack((cos, -sin, sin, cos), -1).view(*freqs.shape, 2, 2)

        # emb = torch.cat((freqs, freqs), -1)

        # self.cos_cache = emb.cos().to(dtype)
        # self.sin_cache = emb.sin().to(dtype)

    def cast_cache(self, device):
        self.freqs_cis = self.freqs_cis.to(device)

        if self.use_complex:
            self.freqs_cis = self.freqs_cis.to(device)

            return

        # self.cos_cache = self.cos_cache.to(device)
        # self.sin_cache = self.sin_cache.to(device)

    def forward(self, position_ids, seq_len=None, device=None, dtype=None):
        if seq_len > self.max_seq_len_cached:
            self.init_cache(seq_len=seq_len, device=device, dtype=dtype)

        self.cast_cache(position_ids.device)

        if self.use_complex:
            freqs_cis = self.freqs_cis[position_ids].unsqueeze(-2)

            return freqs_cis

        return self.freqs_cis[position_ids].unsqueeze(-4)

        # cos = self.cos_cache[position_ids].unsqueeze(-2).to(device=device, dtype=dtype)
        # sin = self.sin_cache[position_ids].unsqueeze(-2).to(device=device, dtype=dtype)

        # return cos, sin


class Llama3RoPE(RotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=8192,
        base=500000,
        use_scaled_rope=False,
        scale_factor=8,
        low_freq_factor=1,
        high_freq_factor=4,
        original_context_len=8192,
        device=None,
        use_complex=False,
    ):
        self.use_scaled_rope = use_scaled_rope
        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_context_len = original_context_len

        super().__init__(dim, max_position_embeddings, base, device, use_complex)

    def get_inv_freq(self):
        inv_freq = super().get_inv_freq()

        if self.use_scaled_rope:
            low_freq_wavlen = self.original_context_len / self.low_freq_factor
            high_freq_wavlen = self.original_context_len / self.high_freq_factor

            wavlen = 2 * math.pi / inv_freq
            inv_freq_llama = torch.where(
                wavlen > low_freq_wavlen, inv_freq / self.scale_factor, inv_freq
            )
            smooth_factor = (
                self.original_context_len / wavlen - self.low_freq_factor
            ) / (self.high_freq_factor - self.low_freq_factor)
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * inv_freq_llama / self.scale_factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavlen < high_freq_wavlen) * ~(wavlen > low_freq_wavlen)
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq


def rotate_half(input):
    x1 = input[..., : input.shape[-1] // 2]
    x2 = input[..., input.shape[-1] // 2 :]

    return torch.cat((-x2, x1), -1)


def apply_rotary_emb_complex(query, key, pos_embed):
    query_c = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    key_c = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    query_c = torch.view_as_real(query_c * pos_embed).flatten(-2)
    key_c = torch.view_as_real(key_c * pos_embed).flatten(-2)

    return query_c.type_as(query), key_c.type_as(key)


def apply_rotary_emb(query, key, pos_embed, use_fp32=False, use_complex=False):
    if use_complex:
        return apply_rotary_emb_complex(query, key, pos_embed)

    # cos, sin = pos_embed
    dtype = query.dtype

    if use_fp32:
        query = query.to(torch.float32)
        key = key.to(torch.float32)

    # cos = cos.to(query.dtype)
    # sin = sin.to(query.dtype)
    pos_embed = pos_embed.to(query.dtype)

    query = query.view(*query.shape[:-1], -1, 1, 2)
    key = key.view(*key.shape[:-1], -1, 1, 2)

    q_embed = (query * pos_embed).sum(-1).flatten(-2)
    k_embed = (key * pos_embed).sum(-1).flatten(-2)

    # q_embed = (query * cos) + (rotate_half(query) * sin)
    # k_embed = (key * cos) + (rotate_half(key) * sin)

    return q_embed.to(dtype), k_embed.to(dtype)
