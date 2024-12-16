from copy import deepcopy
import math
from functools import partial

from slickconf import field, config_fn, call
import torch
from torch import nn

from halite.nn.activation import SwiGLU
from halite.nn.normalization import RMSNorm
from halite.transformers.attention import (
    Attention,
    SelfAttention,
    SelfAttentionQKV,
    DiffAttention,
)
from halite.transformers.block import TransformerEncoderBlock, ResidualBlock
from halite.transformers.embedding import TextEmbedding
from halite.transformers.feedforward import (
    FeedForward,
    GatedFeedForward,
    VocabParallelLinear,
    SequenceParallelWrapper,
)
from halite.transformers.position import RotaryEmbedding, apply_rotary_emb
from halite.transformers.transformer import TransformerDecoder


@config_fn
def build_block(
    layer_id,
    dim,
    n_heads,
    n_layers,
    intermediate_size,
    softcap=0.0,
    rms_norm_epsilon=1e-6,
    post_norm=False,
    attention_processor="native",
    qkv_split=False,
    gated_ff_split=False,
    fast_norm=False,
):
    attention = DiffAttention(
        n_heads,
        dim // n_heads // 2,
        lambda_init=0.8 - 0.6 * call[math.exp](-0.3 * layer_id),
        sub_norm=RMSNorm(
            dim // n_heads * 2, eps=1e-5, elementwise_affine=False, fast=fast_norm
        ),
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
            q_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            k_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            v_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            out_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        )

    else:
        self_attention = SelfAttention(
            qkv=nn.Linear(dim, dim * 3, bias=False),
            attention=attention,
            out=nn.Linear(dim, dim, bias=False),
            qkv_split="llama",
            qkv_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            out_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        )

    if gated_ff_split:
        ff = GatedFeedForward(
            nn.Linear(dim, intermediate_size, bias=False),
            nn.Linear(dim, intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(intermediate_size, dim, bias=False),
            linear_proj_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            linear_gate_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            linear_out_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        )

    else:
        ff = FeedForward(
            nn.Linear(dim, intermediate_size * 2, bias=False),
            SwiGLU(),
            nn.Linear(intermediate_size, dim, bias=False),
            linear1_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            linear2_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
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

    return block


@config_fn
def diff_transformer(
    vocab_size,
    dim,
    n_heads,
    n_layers,
    intermediate_size,
    max_position_embeddings,
    softcap=0.0,
    rms_norm_epsilon=1e-6,
    post_norm=False,
    attention_processor="native",
    qkv_split=False,
    gated_ff_split=False,
):
    blocks = []

    fast_norm = attention_processor == "flash_attn"

    for i in range(n_layers):
        blocks += [
            call[build_block](
                i,
                dim,
                n_heads,
                n_layers,
                intermediate_size,
                softcap,
                rms_norm_epsilon,
                post_norm,
                attention_processor,
                qkv_split,
                gated_ff_split,
                fast_norm,
            )
        ]

    transformer = TransformerDecoder(
        embedding=TextEmbedding(
            nn.Embedding(vocab_size, dim),
            0,
            embed_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            multiplier=dim**0.5,
        ),
        pos_embed=RotaryEmbedding(dim // n_heads // 2, max_position_embeddings),
        blocks=blocks,
        post_blocks=SequenceParallelWrapper(
            RMSNorm(dim, eps=rms_norm_epsilon, fast=fast_norm)
        ),
        head=VocabParallelLinear(
            nn.Linear(dim, vocab_size, bias=False),
            linear_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        ),
        tie_embeds=False,
        use_position_ids=True,
        attention_processor=attention_processor,
    )

    return transformer
