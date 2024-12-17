from copy import deepcopy
from functools import partial

from slickconf import config_fn, call, annotate
from torch import nn

from halite.nn.activation import SwiGLU
from halite.nn.normalization import RMSNorm
from halite.transformers.attention import Attention, SelfAttention, SelfAttentionQKV
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

from halite.transformers.infer.transformer import InferTransformerDecoder
from halite.transformers.infer.block import InferTransformerEncoderBlock

try:
    from halite.transformers.infer.attention import (
        InferSelfAttention,
        InferSelfAttentionQKV,
        FlashInferAttention,
    )

except ImportError:
    pass


@config_fn
def build_block(
    layer_id,
    dim,
    n_heads,
    head_dim,
    n_layers,
    intermediate_size,
    n_key_value_heads=None,
    softcap=0.0,
    rms_norm_epsilon=1e-6,
    post_norm=False,
    pos_embed_apply_fn=None,
    attention_processor="auto",
    qkv_split=False,
    gated_ff_split=False,
    fast_norm=False,
    infer: str | None = None,
):
    if infer == "flashinfer":
        attention = FlashInferAttention(
            annotate("layer_id", layer_id),
            n_heads,
            head_dim,
            n_key_value_heads=n_key_value_heads,
            apply_pos_emb_fn=pos_embed_apply_fn,
        )

    else:
        attention = Attention(
            n_heads,
            head_dim,
            n_key_value_heads=n_key_value_heads,
            attn_drop=0,
            apply_pos_emb_fn=pos_embed_apply_fn,
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

    if n_key_value_heads is None:
        n_key_value_heads = n_heads

    if qkv_split:
        self_attention_class = (
            InferSelfAttentionQKV if infer == "flashinfer" else SelfAttentionQKV
        )

        self_attention = self_attention_class(
            q=nn.Linear(dim, head_dim * n_heads, bias=False),
            k=nn.Linear(dim, head_dim * n_key_value_heads, bias=False),
            v=nn.Linear(dim, head_dim * n_key_value_heads, bias=False),
            attention=attention,
            out=nn.Linear(dim, dim, bias=False),
            q_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            k_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            v_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            out_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        )

    else:
        self_attention_class = (
            InferSelfAttention if infer == "flashinfer" else SelfAttention
        )

        self_attention = self_attention_class(
            qkv=nn.Linear(
                dim, head_dim * (n_heads + n_key_value_heads * 2), bias=False
            ),
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

    if infer is not None:
        block_class = InferTransformerEncoderBlock

    else:
        block_class = TransformerEncoderBlock

    block = block_class(
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
def transformer(
    vocab_size,
    dim,
    n_heads,
    head_dim,
    n_layers,
    intermediate_size,
    context_len,
    n_key_value_heads=None,
    pos_embed=None,
    pos_embed_apply_fn=None,
    softcap=0.0,
    rms_norm_epsilon=1e-6,
    post_norm=False,
    attention_processor="auto",
    qkv_split=False,
    gated_ff_split=False,
    infer: str | None = None,
):
    blocks = []

    fast_norm = attention_processor == "flash_attn"

    if pos_embed is None:
        pos_embed = RotaryEmbedding(head_dim, context_len)

    if pos_embed_apply_fn is None:
        pos_embed_apply_fn = partial(apply_rotary_emb)

    for i in range(n_layers):
        blocks += [
            call[build_block](
                i,
                dim,
                n_heads,
                head_dim,
                n_layers,
                intermediate_size,
                n_key_value_heads,
                softcap,
                rms_norm_epsilon,
                post_norm,
                pos_embed_apply_fn,
                attention_processor,
                qkv_split,
                gated_ff_split,
                fast_norm,
                infer=infer,
            )
        ]

    if infer is not None:
        transformer_class = InferTransformerDecoder

    else:
        transformer_class = TransformerDecoder

    transformer = transformer_class(
        embedding=TextEmbedding(
            nn.Embedding(vocab_size, dim),
            0,
            embed_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        ),
        pos_embed=pos_embed,
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
