import math

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

    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device

        self.init_weights()

    def init_weights(self):
        inv_freq = 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device="cpu")
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        self.register_cos_sin_cache(
            self.max_position_embeddings, "cpu", torch.get_default_dtype()
        )

    def register_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), -1)

        self.register_buffer("cos_cache", emb.cos().to(dtype), persistent=True)
        self.register_buffer("sin_cache", emb.sin().to(dtype), persistent=True)

    def forward(self, position_ids, seq_len=None, device=None, dtype=None):
        if seq_len > self.max_seq_len_cached:
            self.register_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        cos = self.cos_cache[position_ids].unsqueeze(2).to(device=device, dtype=dtype)
        sin = self.sin_cache[position_ids].unsqueeze(2).to(device=device, dtype=dtype)

        return cos, sin


def rotate_half(input):
    x1 = input[..., : input.shape[-1] // 2]
    x2 = input[..., input.shape[-1] // 2 :]

    return torch.cat((-x2, x1), -1)


def apply_rotary_emb(query, key, pos_embed):
    cos, sin = pos_embed
    cos = cos.to(query.dtype)
    sin = sin.to(query.dtype)

    q_embed = (query * cos) + (rotate_half(query) * sin)
    k_embed = (key * cos) + (rotate_half(key) * sin)

    return q_embed, k_embed
