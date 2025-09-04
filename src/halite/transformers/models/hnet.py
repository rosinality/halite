from dataclasses import dataclass
from typing import Any

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
import torch
from torch import nn
from torch.nn import functional as F

from halite.nn.normalization import RMSNorm
from halite.transformers.generation import GenerationMixin
from halite.transformers.initialize import init_weights
from halite.transformers.model import ModelMixin
from halite.transformers.types import UnpadParams, TransformerDecoderOutput


@dataclass
class RouterOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_prob: torch.Tensor


class Router(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)

    def init_weights(self):
        nn.init.eye_(self.query.weight)
        nn.init.eye_(self.key.weight)

    def forward(self, input: torch.Tensor, cu_seqlens: torch.Tensor | None = None):
        query = F.normalize(self.query(input[:, :-1]), dim=-1)
        key = F.normalize(self.key(input[:, 1:]), dim=-1)

        value = torch.einsum("b l d, b l d -> b l", query, key)
        prob = 0.5 * (1 - value)
        prob = F.pad(prob, (1, 0), "constant", 1.0)

        if cu_seqlens is not None:
            prob = prob.squeeze(0)
            prob[cu_seqlens[:-1]] = 1.0

        boundary = prob > 0.5
        selected_prob = torch.where(boundary, prob, 1 - prob)

        return RouterOutput(prob, boundary, selected_prob)


def chunk(
    input: torch.Tensor,
    boundary_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
):
    output = input.squeeze(0)[boundary_mask]
    next_cu_seqlens = F.pad(boundary_mask.cumsum(0)[cu_seqlens[1:] - 1], (1, 0))
    next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())

    return output, next_cu_seqlens, next_max_seqlen


def get_seq_idx(cu_seqlens: torch.Tensor, device: torch.device | None = None):
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.int64, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (seq_idx.cumsum(0) - 1).unsqueeze(0).to(torch.int32)

    return seq_idx


def get_pos_idx(cu_seqlens: torch.Tensor, device: torch.device | None = None):
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.int64, device=device)
    seq_idx[cu_seqlens[:-1]] = cu_seqlens[:-1]
    seq_idx = seq_idx.cumsum(0).unsqueeze(0).to(torch.int32)

    if cu_seqlens.shape[0] > 2:
        seq_diff = torch.zeros_like(seq_idx)
        seq_diff[cu_seqlens[:-1]] = F.pad(cu_seqlens[:-2], (1, 0))
        seq_idx = seq_idx - seq_diff.cumsum(0).unsqueeze(0).to(torch.int32)

    pos_idx = torch.arange(cu_seqlens[-1], device=device).unsqueeze(0).to(torch.int32)
    pos_idx = pos_idx - seq_idx

    return pos_idx


def dechunk(
    input: torch.Tensor,
    boundary_mask: torch.Tensor,
    boundary_prob: torch.Tensor,
    cu_seqlens: torch.Tensor,
    n_heads: int,
    block_size: int,
    dtype: torch.dtype,
):
    prob = torch.clamp(boundary_prob, min=1e-4, max=1 - 1e-4)
    prob = prob[boundary_mask].unsqueeze(0)
    seq_idx = get_seq_idx(cu_seqlens, device=input.device)

    input_dtype = input.dtype

    dt = torch.log(1 / (1 - prob)).unsqueeze(-1).to(dtype)
    x = (input / dt).to(dtype)
    A = -torch.ones((n_heads,), device=input.device, dtype=torch.float32)
    b = prob.to(dtype)
    c = torch.ones_like(b)

    batch, length, dim = x.shape
    head_dim = dim // n_heads

    out = mamba_chunk_scan_combined(
        x.reshape(batch, length, n_heads, head_dim),
        dt.expand(batch, length, n_heads),
        A,
        b.reshape(batch, length, 1, 1),
        c.reshape(batch, length, 1, 1),
        chunk_size=block_size,
        seq_idx=seq_idx,
    )
    out = out.reshape(batch, length, -1)

    plug_back_idx = boundary_mask.cumsum(0) - 1
    out = torch.gather(
        out, dim=1, index=plug_back_idx.reshape(1, -1, 1).expand(-1, -1, dim)
    )

    return out.to(input_dtype)


def ratio_loss(
    target_ratio: float, boundary_mask: torch.Tensor, boundary_prob: torch.Tensor
):
    selected_fraction = boundary_mask.to(torch.float32).mean()
    mean_prob = boundary_prob.to(torch.float32).mean()

    loss = (target_ratio / (target_ratio - 1)) * (
        (target_ratio - 1) * selected_fraction * mean_prob
        + (1 - selected_fraction) * (1 - mean_prob)
    )

    return loss


class Mamba2Block(nn.Module):
    def __init__(self, block: nn.Module, in_proj_init=None, out_proj_init=None):
        super().__init__()

        self.block = block

        self.in_proj_init = in_proj_init
        self.out_proj_init = out_proj_init

    def init_weights(self):
        init_weights(self.in_proj, self.in_proj_init)
        init_weights(self.out_proj, self.out_proj_init)

    def forward(self, input: torch.Tensor, seq_idx: torch.Tensor):
        return self.block(input, seq_idx=seq_idx)


class HNetBlock(nn.Module):
    def __init__(self, dim: int, blocks: list[nn.Module]):
        super().__init__()

        self.blocks = nn.ModuleList(blocks)

        self.norm = RMSNorm(dim, eps=1e-5)

    def forward(
        self, input: torch.Tensor, pos_emb: Any, unpad_params: UnpadParams | None = None
    ):
        seq_idx = get_seq_idx(unpad_params.cu_seqlens_q, device=input.device)

        out = input
        for block in self.blocks:
            if isinstance(block.module, Mamba2Block):
                out = block(out, seq_idx=seq_idx)

            else:
                out = block(out, pos_emb=pos_emb, unpad_params=unpad_params)

        out = self.norm(out)

        return out


class StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.ones_like(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def straight_through(input: torch.Tensor):
    return StraightThrough.apply(input)


class HNet(nn.Module):
    def __init__(
        self,
        dim: int,
        stage_idx: int,
        main_network: nn.Module,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        pos_embed: nn.Module | None = None,
        prev_dim: int | None = None,
    ):
        super().__init__()

        self.stage_idx = stage_idx

        self.encoder = encoder
        self.main_network = main_network
        self.decoder = decoder

        self.pos_embed = pos_embed
        self.pos_embed_attention_bias = getattr(pos_embed, "attention_bias", False)
        self.pos_embed_layer_shared = getattr(pos_embed, "layer_shared", True)

        self.innermost = encoder is None

        if not self.innermost:
            self.router = Router(dim)
            self.residual_proj = nn.Linear(dim, dim, bias=False)

        self.pad_dim = None

        if stage_idx > 0 and dim - prev_dim > 0:
            self.pad_dim = nn.Parameter(torch.zeros(dim - prev_dim))

    def init_weights(self, device):
        def init_weight(module):
            if hasattr(module, "init_weights"):
                module.init_weights()

        for child in self.children():
            child.apply(init_weight)

        self.init_buffers(device)

    def init_buffers(self, device):
        def init_buffer(module):
            if hasattr(module, "init_buffers"):
                module.init_buffers(device)

        for child in self.children():
            child.apply(init_buffer)

    def get_pos_embed(
        self,
        pos_embed,
        attention_mask,
        query_length,
        position_ids,
        unpad_params: UnpadParams | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if self.pos_embed_attention_bias:
            out = pos_embed(attention_mask)

        else:
            out = pos_embed(position_ids, query_length, unpad_params, device, dtype)

        return out

    def residual_fn(
        self, out: torch.Tensor, residual: torch.Tensor, selected_prob: torch.Tensor
    ):
        return out * straight_through(selected_prob) + residual

    def forward(
        self,
        input: torch.Tensor,
        unpad_params: UnpadParams | None = None,
    ):
        pos_emb = None
        if self.pos_embed is not None and self.pos_embed_layer_shared:
            pos_emb = self.get_pos_embed(
                self.pos_embed,
                None,
                unpad_params.max_length_q,
                None,
                unpad_params,
                input.device,
                input.dtype,
            )

        if self.pad_dim is not None:
            input = torch.cat(
                (input, self.pad_dim.expand(input.shape[:-1] + (-1,))), -1
            )

        input_dim = input.shape[-1]

        if self.innermost:
            out = self.main_network(input, pos_emb=pos_emb, unpad_params=unpad_params)

            return out[..., :input_dim], ()

        out = self.encoder(input, pos_emb=pos_emb, unpad_params=unpad_params)

        residual = self.residual_proj(out)

        router_out = self.router(out, cu_seqlens=unpad_params.cu_seqlens_q)

        out, next_cu_seqlens, next_max_seqlen = chunk(
            out, router_out.boundary_mask, cu_seqlens=unpad_params.cu_seqlens_q
        )

        next_unpad_params = UnpadParams(next_cu_seqlens, next_max_seqlen)

        pos_emb_chunked = None
        if self.pos_embed is not None and self.pos_embed_layer_shared:
            pos_emb_chunked = self.get_pos_embed(
                self.pos_embed,
                None,
                next_max_seqlen,
                None,
                next_unpad_params,
                out.device,
                out.dtype,
            )

        out, prev_boundary = self.main_network(
            out,
            pos_emb=pos_emb_chunked,
            unpad_params=next_unpad_params,
        )

        out = dechunk(
            out,
            router_out.boundary_mask,
            router_out.boundary_prob,
            next_cu_seqlens,
            self.n_heads,
            self.block_size,
            self.dtype,
        )

        out = self.residual_fn(out, residual, router_out.selected_prob)

        out = self.decoder(out, pos_emb=pos_emb, unpad_params=unpad_params)

        out = out[..., :input_dim]

        return out, (router_out, *prev_boundary)


class HNetDecoder(nn.Module, GenerationMixin, ModelMixin):
    def __init__(self, embedding, hnet: HNet, post_blocks=None, head=None, config=None):
        super().__init__()

        self.embedding = embedding
        self.hnet = hnet
        self.post_blocks = post_blocks
        self.head = head

        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        unpad_params: UnpadParams | None = None,
    ):
        out = self.embedding(input_ids=input_ids)

        batch, length, _ = out.shape

        if unpad_params is None:
            out = out.reshape(1, batch * length, -1)

            cu_seqlens = torch.arange(batch + 1, device=out.device) * length
            max_seqlen = length

            unpad_params = UnpadParams(cu_seqlens, max_seqlen)

        out, router_outs = self.hnet(out, unpad_params=unpad_params)

        out = out.view(batch, length, -1)

        if self.post_blocks is not None:
            out = self.post_blocks(out)

        out = self.head(out)

        return TransformerDecoderOutput(logits=out, aux_outputs=router_outs)
