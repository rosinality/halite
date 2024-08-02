from copy import deepcopy
from functools import partial

from slickconf import field, config_fn, call
from torch import nn

from halite.transformers.normalization import Scale, L2Norm, LinearLayer, NormalizedLinear, NormalizedEmbedding
from halite.transformers.attention import Attention, SelfAttentionQKV
from halite.transformers.block import TransformerEncoderBlock, NGPTBlock
from halite.transformers.embedding import TextEmbedding
from halite.transformers.feedforward import (
    GatedFeedForward,
    VocabParallelLinear,
)
from halite.transformers.position import RotaryEmbedding, apply_rotary_emb
from halite.transformers.transformer import (
    TransformerConfig,
    TransformerDecoder,
)


@config_fn
def ngpt(
    vocab_size,
    dim,
    n_head,
    n_layer,
    intermediate_size,
    max_position_embeddings,
    softcap=0.0,
    attention_processor="native",
):
    blocks = []

    attention = Attention(
        n_head,
        dim // n_head,
        attn_drop=0,
        apply_pos_emb_fn=partial(apply_rotary_emb),
        processor=attention_processor,
        softcap=softcap,
        is_causal=True,
        normalize=dim**0.5,
    )
    l2norm = L2Norm()
    block = TransformerEncoderBlock(
        NGPTBlock(
            SelfAttentionQKV(
                # q=LinearLayer(nn.Linear(dim, dim, bias=False), L2Norm()),
                # k=LinearLayer(nn.Linear(dim, dim, bias=False), L2Norm()),
                q=nn.Linear(dim, dim, bias=False),
                k=nn.Linear(dim, dim, bias=False),
                v=nn.Linear(dim, dim, bias=False),
                attention=attention,
                out=nn.Linear(dim, dim, bias=False),
                scaler=Scale(dim, 1, 1 / (dim**0.5)),
                q_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
                k_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
                v_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
                out_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            ),
            post_norm=l2norm,
            skip_norm=l2norm,
            scale=Scale(dim, 1 / n_layer, 1 / (dim**0.5)),
        ),
        NGPTBlock(
            GatedFeedForward(
                LinearLayer(
                    nn.Linear(dim, intermediate_size, bias=False),
                    Scale(intermediate_size, 1, 1),
                ),
                LinearLayer(
                    nn.Linear(dim, intermediate_size, bias=False),
                    Scale(intermediate_size, 1, 1, dim**0.5),
                ),
                nn.SiLU(),
                nn.Linear(intermediate_size, dim, bias=False),
                linear_proj_init=partial(
                    nn.init.kaiming_normal_, nonlinearity="linear"
                ),
                linear_gate_init=partial(
                    nn.init.kaiming_normal_, nonlinearity="linear"
                ),
                linear_out_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            ),
            post_norm=l2norm,
            skip_norm=l2norm,
            scale=Scale(dim, 1 / n_layer, 1 / (dim**0.5)),
        ),
    )

    for _ in range(n_layer):
        blocks += [deepcopy(block)]

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
        pos_embed=RotaryEmbedding(dim // n_head, max_position_embeddings),
        blocks=blocks,
        head=VocabParallelLinear(
            LinearLayer(
                nn.Linear(dim, vocab_size, bias=False),
                Scale(vocab_size, 1, 1 / (dim**0.5)),
            ),
            linear_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        ),
        tie_embeds=False,
        use_position_ids=True,
        attention_processor=attention_processor,
        config=transformer_config,
    )

    return transformer


conf = field(model=call[ngpt](32000, 96, 4, 3, call[int](96 * 3.5), 2048, 1e-6))
