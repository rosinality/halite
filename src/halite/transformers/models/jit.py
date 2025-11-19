import math

import torch
from torch.distributed.tensor import distribute_tensor, DTensor
from torch import nn

from halite.transformers.container import ModuleDict
from halite.transformers.initialize import init_weights


def modulate(input, shift, scale):
    return input * (1 + scale) + shift


def init_normal(std=0.02):
    def init(module):
        return nn.init.normal_(module, std=std)

    return init


class TimeEmbedding(nn.Module):
    def __init__(
        self, dim, freq_embed_dim=256, embed_type="positional", use_mlp=True, scale=1
    ):
        super().__init__()

        self.dim = dim

        self.mlp = None
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(freq_embed_dim, dim, bias=True),
                nn.SiLU(),
                nn.Linear(dim, dim, bias=True),
            )

        self.freq_embed_dim = freq_embed_dim
        self.embed_type = embed_type

        if self.embed_type == "fourier":
            self.register_buffer("freqs", torch.randn(freq_embed_dim // 2) * scale)

    def init_weights(self):
        if self.mlp is None:
            return

        init_weights(self.mlp[0], init_normal(std=0.02))
        init_weights(self.mlp[2], init_normal(std=0.02))

    def positional_embed(self, t, max_period=10000):
        half = self.freq_embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float64, device=t.device)
            / half
        )
        args = t[:, None].to(torch.float64) * freqs[None]
        embed = torch.cat((torch.cos(args), torch.sin(args)), -1)

        if self.freq_embed_dim % 2 != 0:
            embed = torch.cat((embed, torch.zeros_like(embed[:, :1])), -1)

        return embed

    def fourier_embed(self, t):
        out = t.to(torch.float64).ger((2 * math.pi * self.freqs.to(torch.float64)))
        out = torch.cat((torch.cos(out), torch.sin(out)), 1)

        return out

    def forward(self, t):
        if self.embed_type == "positional":
            out = self.positional_embed(t, self.dim)

        elif self.embed_type == "fourier":
            out = self.fourier_embed(t)

        else:
            raise ValueError(f"Unknown embed type: {self.embed_type}")

        out = out.to(t.dtype)

        if self.mlp is not None:
            out = self.mlp(out)

        return out


class LabelEmbedding(nn.Module):
    def __init__(self, n_labels, dim):
        super().__init__()

        self.n_labels = n_labels
        self.embed = nn.Embedding(n_labels + 1, dim)

    def init_weights(self):
        init_weights(self.embed, init_normal(std=0.02))

    def forward(self, condition: torch.Tensor, is_uncond: torch.Tensor | None = None):
        if is_uncond is not None:
            uncond_class = condition.new_full((condition.shape[0],), self.n_classes)
            condition = torch.where(is_uncond, uncond_class, condition)

        out = self.embed(condition)

        return out


def get_1d_sincos_pos_embed(dim, pos):
    omega = torch.arange(dim // 2, dtype=torch.float64)
    omega /= dim / 2
    omega = 1 / (10000**omega)

    pos = pos.reshape(-1)
    out = torch.outer(pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat((emb_sin, emb_cos), 1)

    return emb


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim, grid_size, max_position_embeddings):
        super().__init__()

        self.dim = dim
        self.grid_size = grid_size

        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_position_embeddings, dim), requires_grad=False
        )

    def init_weights(self):
        grid_h = torch.arange(self.grid_size, dtype=torch.float32)
        grid_w = torch.arange(self.grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, 0)
        grid0, grid1 = grid.reshape(2, 1, self.grid_size, self.grid_size).unbind(0)

        emb_h = get_1d_sincos_pos_embed(self.dim // 2, grid0)
        emb_w = get_1d_sincos_pos_embed(self.dim // 2, grid1)
        emb = torch.cat([emb_h, emb_w], dim=1).to(torch.float32).unsqueeze(0)

        pos_embed = self.pos_embed.detach()
        if isinstance(pos_embed, DTensor):
            emb = distribute_tensor(
                emb, device_mesh=pos_embed.device_mesh, placements=pos_embed.placements
            )

        pos_embed.copy_(emb)

    def forward(self, input):
        return input + self.pos_embed


def rotate_half(input):
    x1 = input[..., : input.shape[-1] // 2]
    x2 = input[..., input.shape[-1] // 2 :]

    return torch.cat((-x2, x1), -1)


class VisionRoPE(nn.Module):
    def __init__(self, dim, max_position_embeddings, n_label_tokens=0, theta=10000):
        super().__init__()

        self.dim = dim
        self.theta = theta
        self.max_position_embeddings = max_position_embeddings
        self.n_label_tokens = n_label_tokens

        self.register_buffer(
            "freqs_cos",
            torch.empty(max_position_embeddings**2 + n_label_tokens, 1, dim * 2),
        )
        self.register_buffer(
            "freqs_sin",
            torch.empty(max_position_embeddings**2 + n_label_tokens, 1, dim * 2),
        )

    def init_weights(self):
        freqs = 1 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2)[: self.dim // 2].to(torch.float32)
                / self.dim
            )
        )

        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=1)

        freqs1 = freqs.unsqueeze(1)
        freqs2 = freqs.unsqueeze(0)
        freqs = torch.cat(
            (
                freqs1.expand(-1, freqs.shape[0], -1),
                freqs2.expand(freqs.shape[0], -1, -1),
            ),
            -1,
        )

        freqs_flat = freqs.view(-1, freqs.shape[-1])

        if self.n_label_tokens > 0:
            cos_emb = torch.cos(freqs_flat)
            sin_emb = torch.sin(freqs_flat)

            _, embed_dim = cos_emb.shape
            cos_pad = torch.ones(self.n_label_tokens, embed_dim)
            sin_pad = torch.zeros(self.n_label_tokens, embed_dim)

            freqs_cos = torch.cat((cos_pad, cos_emb), 0)
            freqs_sin = torch.cat((sin_pad, sin_emb), 0)

        else:
            freqs_cos = torch.cos(freqs_flat)
            freqs_sin = torch.sin(freqs_flat)

        self.freqs_cos.copy_(freqs_cos.unsqueeze(1))
        self.freqs_sin.copy_(freqs_sin.unsqueeze(1))

    def forward(self):
        return self.freqs_cos, self.freqs_sin


def apply_vision_rope(query, key, pos_embed):
    freqs_cos, freqs_sin = pos_embed

    query = query * freqs_cos + rotate_half(query) * freqs_sin
    key = key * freqs_cos + rotate_half(key) * freqs_sin

    return query, key


class InContextEmbedding(nn.Module):
    def __init__(self, dim, n_tokens):
        super().__init__()

        self.n_tokens = n_tokens

        self.in_context_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))

    def init_weights(self):
        nn.init.normal_(self.in_context_embed, std=0.02)

    def forward(self, input, labels):
        labels = labels.unsqueeze(1).repeat(1, self.n_tokens, 1)
        labels = labels + self.in_context_embed

        return torch.cat((labels, input), 1)


class BottleneckEmbedding(nn.Module):
    def __init__(self, dim, out_dim, linear_init=None):
        super().__init__()

        self.linear = nn.Linear(dim, out_dim)
        self.linear_init = linear_init

    def init_weights(self):
        init_weights(self.linear, self.linear_init)

    def forward(self, input):
        return self.linear(input)


class JiTHead(nn.Module):
    def __init__(self, norm, modulator, dim, out_dim, patch_size):
        super().__init__()

        self.out_dim = out_dim
        self.patch_size = patch_size

        self.norm = norm
        self.modulator = modulator
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)

    def init_weights(self):
        init_weights(self.linear, nn.init.zeros_)
        nn.init.ones_(self.norm.weight)

    def forward(self, input, labels):
        shift, scale = self.modulator(labels)
        out = modulate(self.norm(input), shift, scale)
        out = self.linear(out)

        batch, length, _ = input.shape
        height = width = int(math.sqrt(length))

        out = out.reshape(
            batch, height, width, self.patch_size, self.patch_size, self.out_dim
        )
        out = out.permute(0, 5, 1, 3, 2, 4)
        out = out.reshape(
            batch, self.out_dim, height * self.patch_size, width * self.patch_size
        )

        return out


class JiTModulator(nn.Module):
    def __init__(self, dim, n_factors):
        super().__init__()

        self.n_factors = n_factors

        self.activation = nn.SiLU()
        self.linear = nn.Linear(dim, self.n_factors * dim)

    def init_weights(self):
        init_weights(self.linear, nn.init.zeros_)

    def forward(self, input):
        return (
            self.linear(self.activation(input))
            .unsqueeze(1)
            .chunk(self.n_factors, dim=-1)
        )


class JiTBlock(nn.Module):
    def __init__(self, modulator, norm_attn, self_attention, norm_ff, ff):
        super().__init__()

        self.modulator = modulator
        self.norm_attn = norm_attn
        self.self_attention = self_attention
        self.norm_ff = norm_ff
        self.ff = ff

    def init_weights(self):
        nn.init.ones_(self.norm_attn.weight)
        nn.init.ones_(self.norm_ff.weight)

    def forward(self, input, condition, pos_emb=None):
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = self.modulator(
            condition
        )

        out, _ = self.self_attention(
            modulate(self.norm_attn(input), shift_attn, scale_attn), pos_emb=pos_emb
        )

        out = input + gate_attn * out
        out = out + gate_ff * self.ff(modulate(self.norm_ff(out), shift_ff, scale_ff))

        return out


class JiT(nn.Module):
    def __init__(
        self,
        embedding,
        time_embedding,
        label_embedding,
        post_embed=None,
        pos_embed=None,
        pos_rope=None,
        in_context_embed=None,
        in_context_rope=None,
        in_context_start=0,
        blocks=None,
        head=None,
    ):
        super().__init__()

        self.embedding = embedding
        self.time_embedding = time_embedding
        self.label_embedding = label_embedding

        self.post_embed = post_embed
        self.pos_embed = pos_embed
        self.pos_rope = pos_rope
        self.in_context_embed = in_context_embed
        self.in_context_rope = in_context_rope
        self.in_context_start = in_context_start

        self.blocks = ModuleDict()
        blocks = [] if blocks is None else blocks
        for i, block in enumerate(blocks):
            self.blocks[str(i)] = block

        self.head = head

    def init_weights(self, device):
        def init_weight(module):
            if hasattr(module, "init_weights"):
                module.init_weights()

        for child in self.children():
            child.apply(init_weight)

    def forward(self, input, times, labels):
        out = self.embedding(input)

        if self.post_embed is not None:
            out = self.post_embed(out)

        if self.pos_embed is not None:
            out = self.pos_embed(out)

        time_embed = self.time_embedding(times)
        label_embed = self.label_embedding(labels)
        cond_embed = time_embed + label_embed

        total_time = 0
        n_blocks = 0
        for id, block in self.blocks.items():
            id = int(id)

            if self.in_context_embed is not None and id == self.in_context_start:
                out = self.in_context_embed(out, label_embed)

            pos_emb = None

            if id < self.in_context_start and self.pos_rope is not None:
                pos_emb = self.pos_rope()

            elif id >= self.in_context_start and self.in_context_rope is not None:
                pos_emb = self.in_context_rope()

            # torch.cuda.synchronize()
            # start_time = perf_counter()
            out = block(out, cond_embed, pos_emb=pos_emb)
            # torch.cuda.synchronize()
            # end_time = perf_counter()
            # total_time += end_time - start_time
            # n_blocks += 1

        # if dist.get_rank() == 0:
        #     print(f"Average block time: {total_time / n_blocks}")

        if self.in_context_embed is not None:
            out = out[:, self.in_context_embed.n_tokens :]

        if self.head is not None:
            out = self.head(out, cond_embed)

        return out
