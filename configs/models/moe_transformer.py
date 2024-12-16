from copy import deepcopy
from functools import partial

from slickconf import field, config_fn, call
import torch
from torch import nn

from halite.nn.init import kaiming_normal_
from halite.nn.activation import SwiGLU
from halite.nn.normalization import RMSNorm
from halite.transformers.attention import Attention, SelfAttention, SelfAttentionQKV
from halite.transformers.block import TransformerEncoderBlock, ResidualBlock
from halite.transformers.embedding import TextEmbedding
from halite.transformers.feedforward import (
    VocabParallelLinear,
    SequenceParallelWrapper,
)
from halite.transformers.position import RotaryEmbedding, apply_rotary_emb
from halite.transformers.transformer import TransformerDecoder
from halite.transformers.moe.moe import MoE
from halite.transformers.moe.router import TopKRouter
from halite.transformers.moe.scattermoe.scattermoe import ExpertLinear, FeedForward


@config_fn
def moe_transformer(
    vocab_size,
    dim,
    n_heads,
    n_layers,
    n_experts,
    expert_top_k,
    z_loss,
    load_balance_loss,
    intermediate_size,
    max_position_embeddings,
    softcap=0.0,
    rms_norm_epsilon=1e-6,
    post_norm=False,
    attention_processor="native",
    qkv_split=False,
):
    blocks = []

    fast_norm = attention_processor == "flash_attn"
    kaiming_init = partial(kaiming_normal_, nonlinearity="linear", truncate=3)

    attention = Attention(
        n_heads,
        dim // n_heads,
        attn_drop=0,
        apply_pos_emb_fn=partial(apply_rotary_emb),
        processor=attention_processor,
        softcap=softcap,
        is_causal=True,
    )
    rmsnorm = RMSNorm(dim, eps=rms_norm_epsilon, fast=fast_norm)
    rmsnorm_post = RMSNorm(
        dim,
        eps=rms_norm_epsilon,
        weight_init=partial(nn.init.constant_, val=1 / (n_layers**0.5)),
        fast=fast_norm,
    )

    if qkv_split:
        self_attention = SelfAttentionQKV(
            q=nn.Linear(dim, dim, bias=False),
            k=nn.Linear(dim, dim, bias=False),
            v=nn.Linear(dim, dim, bias=False),
            attention=attention,
            out=nn.Linear(dim, dim, bias=False),
            q_init=kaiming_init,
            k_init=kaiming_init,
            v_init=kaiming_init,
            out_init=kaiming_init,
        )

    else:
        self_attention = SelfAttention(
            qkv=nn.Linear(dim, dim * 3, bias=False),
            attention=attention,
            out=nn.Linear(dim, dim, bias=False),
            qkv_split="llama",
            qkv_init=kaiming_init,
            out_init=kaiming_init,
        )

    ff = MoE(
        TopKRouter(
            dim,
            n_experts,
            expert_top_k,
            gate_init=kaiming_init,
            z_loss=z_loss,
            load_balance_loss=load_balance_loss,
            pre_softmax=True,
            deterministic=True,
        ),
        None,
        FeedForward(
            ExpertLinear(n_experts, dim, intermediate_size * 2),
            SwiGLU(),
            ExpertLinear(n_experts, intermediate_size, dim),
            top_k=expert_top_k,
            linear1_init=partial(kaiming_normal_, nonlinearity="linear", truncate=3),
            linear2_init=partial(kaiming_normal_, nonlinearity="linear", truncate=3),
        ),
    )

    block = TransformerEncoderBlock(
        ResidualBlock(
            deepcopy(rmsnorm),
            self_attention,
            post_norm=(deepcopy(rmsnorm_post) if post_norm else None),
        ),
        ResidualBlock(
            deepcopy(rmsnorm),
            ff,
            post_norm=(deepcopy(rmsnorm_post) if post_norm else None),
        ),
    )

    for _ in range(n_layers):
        blocks += [deepcopy(block)]

    transformer = TransformerDecoder(
        embedding=TextEmbedding(
            nn.Embedding(vocab_size, dim),
            0,
            embed_init=kaiming_init,
            multiplier=dim**0.5,
        ),
        pos_embed=RotaryEmbedding(dim // n_heads, max_position_embeddings),
        blocks=blocks,
        post_blocks=SequenceParallelWrapper(
            RMSNorm(dim, eps=rms_norm_epsilon, fast=fast_norm)
        ),
        head=VocabParallelLinear(
            nn.Linear(dim, vocab_size, bias=False),
            linear_init=kaiming_init,
        ),
        tie_embeds=False,
        use_position_ids=True,
        attention_processor=attention_processor,
    )

    return transformer
