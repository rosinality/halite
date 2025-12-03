from copy import deepcopy
from functools import partial

from slickconf import config_fn, call, select
from torch import nn

from halite.nn.activation import SwiGLU, GEGLU
from halite.nn.normalization import RMSNorm
from halite.transformers.attention import Attention, SelfAttention
from halite.transformers.feedforward import (
    FeedForward,
)
from halite.transformers.models.jit import (
    apply_vision_rope,
    JiTBlock,
    JiTModulator,
    EncoderJiT,
    TimeEmbedding,
    LabelEmbedding,
    BottleneckEmbedding,
    SinusoidalEmbedding,
    VisionRoPE,
    InContextEmbedding,
    JiTHead,
)
# from halite.projects.dit.model import JiTBlock, BottleneckPatchEmbed


@config_fn
def get_activation(activation):
    activation_map = {
        "swiglu": (SwiGLU(), nn.SiLU()),
        "gelu_tanh": (nn.GELU(approximate="tanh"), None),
        "geglu_tanh": (GEGLU(approximate="tanh"), nn.GELU(approximate="tanh")),
    }

    return activation_map.get(activation)


@config_fn
def build_block(
    layer_id,
    dim,
    n_heads,
    head_dim,
    intermediate_size,
    n_key_value_heads=None,
    rms_norm_epsilon=1e-6,
    pos_embed_apply_fn=None,
    attention_processor="auto",
    attention_bias=False,
    ffn_bias=False,
    ffn_activation="swiglu",
    fast_norm=False,
    q_norm=None,
    k_norm=None,
):
    attention = Attention(
        n_heads,
        head_dim,
        n_key_value_heads=n_key_value_heads,
        attn_drop=0,
        apply_pos_emb_fn=pos_embed_apply_fn,
        processor=attention_processor,
        is_causal=False,
        q_norm=deepcopy(q_norm),
        k_norm=deepcopy(k_norm),
    )

    norm = RMSNorm(dim, eps=rms_norm_epsilon, fast=fast_norm)

    if n_key_value_heads is None:
        n_key_value_heads = n_heads

    self_attention = SelfAttention(
        qkv=nn.Linear(
            dim, head_dim * (n_heads + n_key_value_heads * 2), bias=attention_bias
        ),
        attention=attention,
        out=nn.Linear(head_dim * n_heads, dim, bias=attention_bias),
        qkv_split="llama",
        qkv_init=partial(nn.init.xavier_uniform_),
        out_init=partial(nn.init.xavier_uniform_),
    )

    gated_act, base_act = call[get_activation](ffn_activation)

    gate_intermediate_size = intermediate_size
    if base_act is not None:
        gate_intermediate_size = intermediate_size * 2

    ff = FeedForward(
        nn.Linear(dim, gate_intermediate_size, bias=ffn_bias),
        deepcopy(gated_act),
        nn.Linear(intermediate_size, dim, bias=ffn_bias),
        linear1_init=partial(nn.init.xavier_uniform_),
        linear2_init=partial(nn.init.xavier_uniform_),
    )

    block = JiTBlock(
        JiTModulator(dim, 6), deepcopy(norm), self_attention, deepcopy(norm), ff
    )

    return block


@config_fn
def encoder_jit(
    image_size,
    patch_size,
    n_labels,
    dim,
    patch_dim,
    n_heads,
    head_dim,
    n_layers,
    intermediate_size,
    in_context_len,
    in_context_start=0,
    n_key_value_heads=None,
    rms_norm_epsilon=1e-6,
    attention_processor="auto",
    attention_bias=True,
    ffn_bias=True,
    ffn_activation="swiglu",
    fast_norm=False,
    qk_norm=False,
    cond_start_id=0,
):
    blocks = []

    time_embedding = TimeEmbedding(dim)
    label_embedding = LabelEmbedding(n_labels, dim)

    grid_size = image_size // patch_size
    rope_dim = head_dim // 2

    in_context_embed = InContextEmbedding(dim, in_context_len)
    in_context_rope = VisionRoPE(rope_dim, grid_size, in_context_len)

    pos_embed_apply_fn = partial(apply_vision_rope)

    norm = RMSNorm(dim, eps=rms_norm_epsilon, fast=fast_norm)

    q_norm = None
    k_norm = None

    head = JiTHead(norm, JiTModulator(dim, 2), dim, 3, patch_size)

    fast_norm = attention_processor == "flash_attn"

    for i in range(n_layers):
        if qk_norm:
            q_norm = RMSNorm(head_dim, eps=rms_norm_epsilon, fast=fast_norm)
            k_norm = deepcopy(q_norm)

        blocks += [
            call[build_block](
                i,
                dim,
                n_heads,
                head_dim,
                intermediate_size,
                n_key_value_heads,
                rms_norm_epsilon,
                pos_embed_apply_fn,
                attention_processor,
                attention_bias,
                ffn_bias,
                ffn_activation,
                fast_norm,
                q_norm=q_norm,
                k_norm=k_norm,
            )
        ]

    transformer = partial(
        EncoderJiT,
        time_embedding=time_embedding,
        label_embedding=label_embedding,
        in_context_embed=in_context_embed,
        in_context_rope=in_context_rope,
        blocks=blocks,
        head=head,
        cond_start_id=cond_start_id,
    )

    return transformer


@config_fn
def use_flash_attention(model_conf):
    conf = (
        select(model_conf).instance(Attention).update_dict(dict(processor="flash_attn"))
    )

    return conf
