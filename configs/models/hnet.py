from copy import deepcopy
from functools import partial

from slickconf import config_fn, call
from mamba_ssm.modules.mamba2 import Mamba2
from torch import nn

from halite.nn.activation import SwiGLU, GEGLU
from halite.nn.init import kaiming_linear_
from halite.nn.normalization import RMSNorm
from halite.transformers.attention import Attention, SelfAttention, SelfAttentionQKV
from halite.transformers.block import TransformerEncoderBlock, ResidualBlock
from halite.transformers.embedding import TextEmbedding
from halite.transformers.feedforward import (
    FeedForward,
    GatedFeedForward,
    VocabParallelLinear,
)
from halite.transformers.position import RotaryEmbedding, apply_rotary_emb
from halite.transformers.models.hnet import HNet, HNetDecoder, HNetBlock, Mamba2Block


@config_fn
def get_activation(activation):
    activation_map = {
        "swiglu": (SwiGLU(), nn.SiLU()),
        "gelu_tanh": (nn.GELU(approximate="tanh"), None),
        "geglu_tanh": (GEGLU(approximate="tanh"), nn.GELU(approximate="tanh")),
    }

    return activation_map.get(activation)


@config_fn
def get_value(map, key, default=None):
    return map.get(key, default)


@config_fn
def get_stage_kwargs(map_or_list, stage_idx):
    if isinstance(map_or_list, dict):
        return map_or_list

    else:
        return map_or_list[stage_idx]


@config_fn
def build_layer(
    layer_id,
    dim,
    intermediate_size,
    n_res_blocks,
    seq_mixer="attention",
    attn_kwargs=None,
    mamba2_kwargs=None,
    use_ffn=True,
    norm=None,
    rms_norm_epsilon=1e-6,
    post_norm=False,
    pos_embed_apply_fn=None,
    attn_bias=False,
    qkv_split=False,
    ffn_bias=False,
    ffn_activation="swiglu",
    gated_ff_split=False,
    fast_norm=False,
    q_norm=None,
    k_norm=None,
    layer_type=None,
):
    attn_kwargs = attn_kwargs or {}
    mamba2_kwargs = mamba2_kwargs or {}

    n_heads = attn_kwargs.get("n_heads", None)
    head_dim = attn_kwargs.get("head_dim", None)
    n_key_value_heads = attn_kwargs.get("n_key_value_heads", None)

    if seq_mixer == "attention":
        mixer = Attention(
            **attn_kwargs,
            apply_pos_emb_fn=pos_embed_apply_fn,
            q_norm=deepcopy(q_norm),
            k_norm=deepcopy(k_norm),
        )

    elif seq_mixer == "mamba2":
        mixer_layer = Mamba2Block(
            Mamba2(
                d_model=dim,
                **mamba2_kwargs,
            ),
            in_proj_init=partial(kaiming_linear_),
            out_proj_init=partial(kaiming_linear_, scale=1 / (n_res_blocks**0.5)),
        )

    norm_post = norm
    if norm is None:
        norm = RMSNorm(dim, eps=rms_norm_epsilon, fast=fast_norm)
        norm_post = RMSNorm(
            dim,
            eps=rms_norm_epsilon,
            weight_init=partial(nn.init.constant_, val=1 / (n_res_blocks**0.5)),
            fast=fast_norm,
        )

    if n_key_value_heads is None:
        n_key_value_heads = n_heads

    if seq_mixer == "attention":
        if qkv_split:
            mixer_layer = SelfAttentionQKV(
                q=nn.Linear(dim, head_dim * n_heads, bias=attn_bias),
                k=nn.Linear(dim, head_dim * n_key_value_heads, bias=attn_bias),
                v=nn.Linear(dim, head_dim * n_key_value_heads, bias=attn_bias),
                attention=mixer,
                out=nn.Linear(head_dim * n_heads, dim, bias=attn_bias),
                q_init=partial(kaiming_linear_),
                k_init=partial(kaiming_linear_),
                v_init=partial(kaiming_linear_),
                out_init=partial(kaiming_linear_, scale=1 / (n_res_blocks**0.5)),
                layer_type=layer_type,
            )

        else:
            mixer_layer = SelfAttention(
                qkv=nn.Linear(
                    dim,
                    head_dim * (n_heads + n_key_value_heads * 2),
                    bias=attn_bias,
                ),
                attention=mixer,
                out=nn.Linear(head_dim * n_heads, dim, bias=attn_bias),
                qkv_split="llama",
                qkv_init=partial(kaiming_linear_),
                out_init=partial(kaiming_linear_, scale=1 / (n_res_blocks**0.5)),
                layer_type=layer_type,
            )

    if use_ffn:
        gated_act, base_act = call[get_activation](ffn_activation)

        if gated_ff_split:
            ff = GatedFeedForward(
                nn.Linear(dim, intermediate_size, bias=ffn_bias),
                nn.Linear(dim, intermediate_size, bias=ffn_bias),
                deepcopy(base_act),
                nn.Linear(intermediate_size, dim, bias=ffn_bias),
                linear_proj_init=partial(kaiming_linear_),
                linear_gate_init=partial(kaiming_linear_),
                linear_out_init=partial(kaiming_linear_, scale=1 / (n_res_blocks**0.5)),
            )

        else:
            gate_intermediate_size = intermediate_size
            if base_act is not None:
                gate_intermediate_size = intermediate_size * 2

            ff = FeedForward(
                nn.Linear(dim, gate_intermediate_size, bias=ffn_bias),
                deepcopy(gated_act),
                nn.Linear(intermediate_size, dim, bias=ffn_bias),
                linear1_init=partial(kaiming_linear_),
                linear2_init=partial(kaiming_linear_, scale=1 / (n_res_blocks**0.5)),
            )

        block = TransformerEncoderBlock(
            ResidualBlock(
                deepcopy(norm),
                mixer_layer,
                post_norm=(deepcopy(norm_post) if post_norm else None),
            ),
            ResidualBlock(
                deepcopy(norm),
                ff,
                post_norm=(deepcopy(norm_post) if post_norm else None),
            ),
        )

    else:
        block = ResidualBlock(
            deepcopy(norm),
            mixer_layer,
            post_norm=(deepcopy(norm_post) if post_norm else None),
        )

    return block


@config_fn
def get_n_res_blocks(layers):
    n_blocks = 0

    for layer in layers:
        n_layer = get_value(layer, "n_layer", 1)
        use_ffn = get_value(layer, "use_ffn", True)

        if use_ffn:
            n_blocks += n_layer * 2

        else:
            n_blocks += n_layer

    return n_blocks


@config_fn
def build_block(
    layer_id,
    layers,
    dim,
    intermediate_size,
    n_res_blocks,
    attn_kwargs=None,
    mamba2_kwargs=None,
    norm=None,
    pos_embed_apply_fn=None,
):
    blocks = []

    for layer in layers:
        mixer_type = get_value(layer, "mixer", "attention")
        n_layer = get_value(layer, "n_layer", 1)
        use_ffn = get_value(layer, "use_ffn", True)

        for _ in range(n_layer):
            blocks += [
                build_layer(
                    layer_id=layer_id,
                    dim=dim,
                    intermediate_size=intermediate_size,
                    n_res_blocks=n_res_blocks,
                    seq_mixer=mixer_type,
                    attn_kwargs=attn_kwargs,
                    mamba2_kwargs=mamba2_kwargs,
                    use_ffn=use_ffn,
                    norm=norm,
                    pos_embed_apply_fn=pos_embed_apply_fn,
                )
            ]

            layer_id += 1

    block = HNetBlock(dim, blocks)

    return block


def build_hnet(
    arch,
    dims,
    intermediate_sizes,
    attn_kwargs,
    mamba2_kwargs,
    norm,
    pos_embeds,
    pos_embed_apply_fn,
    stage_idx=0,
    parent_n_res_blocks=0,
):
    n_res_blocks = parent_n_res_blocks

    if len(arch) == 3:
        encoder, main_network, decoder = arch
        innermost = False

    elif len(arch) == 1:
        main_network = arch[0]
        encoder = None
        decoder = None
        innermost = True

    if innermost:
        return build_block(
            stage_idx,
            main_network,
            dims[stage_idx],
            intermediate_sizes[stage_idx],
            n_res_blocks + get_n_res_blocks(main_network),
            get_stage_kwargs(attn_kwargs, stage_idx),
            get_stage_kwargs(mamba2_kwargs, stage_idx),
            norm,
            pos_embed_apply_fn,
        )

    else:
        n_res_blocks += get_n_res_blocks(encoder)
        n_res_blocks += get_n_res_blocks(decoder)

        encoder = build_block(
            stage_idx,
            encoder,
            dims[stage_idx],
            intermediate_sizes[stage_idx],
            n_res_blocks,
            get_stage_kwargs(attn_kwargs, stage_idx),
            get_stage_kwargs(mamba2_kwargs, stage_idx),
            norm,
            pos_embed_apply_fn,
        )
        decoder = build_block(
            stage_idx,
            decoder,
            dims[stage_idx],
            intermediate_sizes[stage_idx],
            n_res_blocks,
            get_stage_kwargs(attn_kwargs, stage_idx),
            get_stage_kwargs(mamba2_kwargs, stage_idx),
            norm,
            pos_embed_apply_fn,
        )

        return HNet(
            dims[stage_idx],
            stage_idx,
            call[build_hnet](
                [main_network],
                dims,
                intermediate_sizes,
                attn_kwargs,
                mamba2_kwargs,
                norm,
                pos_embeds,
                pos_embed_apply_fn,
                stage_idx + 1,
                n_res_blocks,
            ),
            encoder,
            decoder,
            pos_embeds[stage_idx],
            dims[stage_idx - 1] if stage_idx > 0 else None,
        )


@config_fn
def hnet(
    arch,
    dims,
    intermediate_sizes,
    pos_dims,
    attn_kwargs,
    mamba2_kwargs,
    vocab_size,
    context_len,
    pos_embed="rope",
    pos_embed_apply_fn=None,
    embedding=None,
    use_head=True,
):
    if isinstance(pos_embed, str) and pos_embed == "rope":
        pos_embeds = [
            RotaryEmbedding(pos_dim, context_len, use_fused=True)
            for pos_dim in pos_dims
        ]
        pos_embed_apply_fn = partial(apply_rotary_emb, use_fused=True, inplace=True)

    if embedding is None:
        embedding = TextEmbedding(
            nn.Embedding(vocab_size, dims[0]),
            0,
            embed_init=partial(kaiming_linear_),
        )

    head = None

    if use_head:
        head = VocabParallelLinear(
            nn.Linear(dims[0], vocab_size, bias=False),
            linear_init=partial(kaiming_linear_),
        )

    hnet_model = build_hnet(
        arch,
        dims,
        intermediate_sizes,
        attn_kwargs=attn_kwargs,
        mamba2_kwargs=mamba2_kwargs,
        norm=None,
        pos_embeds=pos_embeds,
        pos_embed_apply_fn=pos_embed_apply_fn,
    )

    hnet_decoder = HNetDecoder(
        embedding=embedding,
        hnet=hnet_model,
        head=head,
    )

    return hnet_decoder
