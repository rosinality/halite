from copy import deepcopy
from functools import partial

from slickconf import field, config_fn, call
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
from halite.transformers.transformer import (
    TransformerConfig,
    TransformerDecoder,
)

try:
    from halite.transformers.infer.attention import (
        InferSelfAttention,
        FlashInferAttention,
    )

except ImportError:
    pass


@config_fn
def build_block(
    layer_id,
    dim,
    n_head,
    n_layer,
    intermediate_size,
    n_key_value_head=None,
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
    head_dim = dim // n_head

    if infer == "flashinfer":
        attention = FlashInferAttention(
            layer_id,
            n_head,
            head_dim,
            apply_pos_emb_fn=pos_embed_apply_fn,
        )

    else:
        attention = Attention(
            n_head,
            head_dim,
            n_key_value_head=n_key_value_head,
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
        weight_init=partial(nn.init.constant_, val=1 / (n_layer**0.5)),
        fast=fast_norm,
    )

    if n_key_value_head is None:
        n_key_value_head = n_head

    if qkv_split:
        self_attention = SelfAttentionQKV(
            q=nn.Linear(dim, head_dim * n_head, bias=False),
            k=nn.Linear(dim, head_dim * n_key_value_head, bias=False),
            v=nn.Linear(dim, head_dim * n_key_value_head, bias=False),
            attention=attention,
            out=nn.Linear(dim, dim, bias=False),
            q_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            k_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            v_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            out_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        )

    else:
        if infer == "flashinfer":
            self_attention = InferSelfAttention(
                qkv=nn.Linear(
                    dim, head_dim * (n_head + n_key_value_head * 2), bias=False
                ),
                attention=attention,
                out=nn.Linear(dim, dim, bias=False),
                qkv_split="llama",
                qkv_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
                out_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            )

        else:
            self_attention = SelfAttention(
                qkv=nn.Linear(
                    dim, head_dim * (n_head + n_key_value_head * 2), bias=False
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
def transformer(
    vocab_size,
    dim,
    n_head,
    n_layer,
    intermediate_size,
    max_position_embeddings,
    n_key_value_head=None,
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
        pos_embed = RotaryEmbedding(dim // n_head, max_position_embeddings)

    if pos_embed_apply_fn is None:
        pos_embed_apply_fn = partial(apply_rotary_emb)

    for i in range(n_layer):
        blocks += [
            call[build_block](
                i,
                dim,
                n_head,
                n_layer,
                intermediate_size,
                n_key_value_head,
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

    transformer_config = TransformerConfig(
        dim=dim,
        n_heads=n_head,
        head_dim=dim // n_head,
        n_heads_tp=n_head,
        max_length=None,
        n_layers=n_layer,
        vocab_size=vocab_size,
    )

    transformer = TransformerDecoder(
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
        config=transformer_config,
    )

    return transformer


conf = field(model=call[transformer](32000, 96, 4, 3, call[int](96 * 3.5), 2048, 1e-6))
