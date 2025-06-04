from copy import deepcopy
from functools import partial

from slickconf import config_fn, call, annotate, patch, patch_fn, select
from torch import nn

from halite.nn.activation import SwiGLU, GEGLU
from halite.nn.normalization import RMSNorm
from halite.transformers.attention import Attention, SelfAttention, SelfAttentionQKV
from halite.transformers.block import TransformerEncoderBlock, ResidualBlock
from halite.transformers.embedding import TextEmbedding
from halite.transformers.feedforward import (
    FeedForward,
    GatedFeedForward,
    VocabParallelLinear,
    SequenceParallelWrapper,
    FusedLinearCrossEntropy,
)
from halite.transformers.position import RotaryEmbedding, apply_rotary_emb
from halite.transformers.transformer import TransformerDecoder
from halite.transformers.infer.transformer import InferTransformerDecoder
from halite.transformers.infer.block import InferTransformerEncoderBlock
from halite.transformers.tokainfer.block import TokaInferTransformerEncoderBlock
from halite.transformers.tokainfer.transformer import TokaInferTransformerDecoder

try:
    from halite.transformers.infer.attention import (
        InferSelfAttention,
        InferSelfAttentionQKV,
        FlashInferAttention,
    )

except ImportError:
    pass

try:
    from halite.transformers.tokainfer.attention import (
        TokaInferSelfAttention,
        TokaInferSelfAttentionQKV,
        TokaInferAttention,
    )

except ImportError:
    pass


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
    n_layers,
    intermediate_size,
    n_key_value_heads=None,
    softcap=0.0,
    norm=None,
    rms_norm_epsilon=1e-6,
    post_norm=False,
    pos_embed_apply_fn=None,
    attention_processor="auto",
    attention_bias=False,
    is_causal=True,
    qkv_split=False,
    ffn_bias=False,
    ffn_activation="swiglu",
    gated_ff_split=False,
    fast_norm=False,
    infer: str | None = None,
    q_norm=None,
    k_norm=None,
    layer_type=None,
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
            is_causal=is_causal,
            q_norm=deepcopy(q_norm),
            k_norm=deepcopy(k_norm),
        )

    norm_post = norm
    if norm is None:
        norm = RMSNorm(dim, eps=rms_norm_epsilon, fast=fast_norm)
        norm_post = RMSNorm(
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
            q=nn.Linear(dim, head_dim * n_heads, bias=attention_bias),
            k=nn.Linear(dim, head_dim * n_key_value_heads, bias=attention_bias),
            v=nn.Linear(dim, head_dim * n_key_value_heads, bias=attention_bias),
            attention=attention,
            out=nn.Linear(head_dim * n_heads, dim, bias=attention_bias),
            q_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            k_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            v_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            out_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            layer_type=layer_type,
        )

    else:
        self_attention_class = (
            InferSelfAttention if infer == "flashinfer" else SelfAttention
        )

        self_attention = self_attention_class(
            qkv=nn.Linear(
                dim, head_dim * (n_heads + n_key_value_heads * 2), bias=attention_bias
            ),
            attention=attention,
            out=nn.Linear(head_dim * n_heads, dim, bias=attention_bias),
            qkv_split="llama",
            qkv_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            out_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            layer_type=layer_type,
        )

    gated_act, base_act = call[get_activation](ffn_activation)

    if gated_ff_split:
        ff = GatedFeedForward(
            nn.Linear(dim, intermediate_size, bias=ffn_bias),
            nn.Linear(dim, intermediate_size, bias=ffn_bias),
            deepcopy(base_act),
            nn.Linear(intermediate_size, dim, bias=ffn_bias),
            linear_proj_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            linear_gate_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            linear_out_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        )

    else:
        gate_intermediate_size = intermediate_size
        if base_act is not None:
            gate_intermediate_size = intermediate_size * 2

        ff = FeedForward(
            nn.Linear(dim, gate_intermediate_size, bias=ffn_bias),
            deepcopy(gated_act),
            nn.Linear(intermediate_size, dim, bias=ffn_bias),
            linear1_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
            linear2_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        )

    if infer is not None:
        block_class = InferTransformerEncoderBlock

    else:
        block_class = TransformerEncoderBlock

    block = block_class(
        ResidualBlock(
            deepcopy(norm),
            self_attention,
            post_norm=(deepcopy(norm_post) if post_norm else None),
        ),
        ResidualBlock(
            deepcopy(norm),
            ff,
            post_norm=(deepcopy(norm_post) if post_norm else None),
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
    pos_embed="rope",
    pos_embed_apply_fn=None,
    softcap=0.0,
    is_causal=True,
    norm=None,
    rms_norm_epsilon=1e-6,
    post_norm=False,
    attention_processor="auto",
    attention_bias=False,
    flex_attention_processor=None,
    qkv_split=False,
    gated_ff_split=False,
    ffn_bias=False,
    ffn_activation="swiglu",
    infer: str | None = None,
    embedding=None,
    q_norm=None,
    k_norm=None,
    use_head=True,
    tie_embeds=False,
    layer_types=None,
):
    blocks = []

    fast_norm = attention_processor == "flash_attn"

    if isinstance(pos_embed, str) and pos_embed == "rope":
        pos_embed = RotaryEmbedding(head_dim, context_len)
        pos_embed_apply_fn = partial(apply_rotary_emb)

    for i in range(n_layers):
        layer_type = None
        if layer_types is not None:
            layer_type = layer_types[i]

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
                norm,
                rms_norm_epsilon,
                post_norm,
                pos_embed_apply_fn,
                attention_processor,
                attention_bias,
                is_causal,
                qkv_split,
                ffn_bias,
                ffn_activation,
                gated_ff_split,
                fast_norm,
                infer=infer,
                q_norm=q_norm,
                k_norm=k_norm,
                layer_type=layer_type,
            )
        ]

    if infer is not None:
        transformer_class = InferTransformerDecoder

    else:
        transformer_class = TransformerDecoder

    if embedding is None:
        embedding = TextEmbedding(
            nn.Embedding(vocab_size, dim),
            0,
            embed_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        )

    head = None

    if use_head:
        head = VocabParallelLinear(
            nn.Linear(dim, vocab_size, bias=False),
            linear_init=partial(nn.init.kaiming_normal_, nonlinearity="linear"),
        )

    if norm is None:
        norm = RMSNorm(dim, eps=rms_norm_epsilon, fast=fast_norm)

    transformer = transformer_class(
        embedding=embedding,
        pos_embed=pos_embed,
        blocks=blocks,
        post_blocks=SequenceParallelWrapper(norm),
        head=head,
        tie_embeds=tie_embeds,
        use_position_ids=True,
        attention_processor=attention_processor,
        flex_attention_processor=flex_attention_processor,
    )

    return transformer


@config_fn
def transformer_infer(
    vocab_size,
    dim,
    n_heads,
    head_dim,
    n_layers,
    intermediate_size,
    context_len,
    n_key_value_heads=None,
    pos_embed="rope",
    pos_embed_apply_fn=None,
    softcap=0.0,
    is_causal=True,
    norm=None,
    rms_norm_epsilon=1e-6,
    post_norm=False,
    attention_processor="auto",
    attention_bias=False,
    flex_attention_processor=None,
    qkv_split=False,
    gated_ff_split=False,
    ffn_bias=False,
    ffn_activation="swiglu",
    infer: str | None = None,
    embedding=None,
    q_norm=None,
    k_norm=None,
    use_head=True,
    tie_embeds=False,
    layer_types=None,
):
    if infer is None:
        return partial(patch, patches=())

    patches = [
        patch_fn.select()
        .instance(Attention)
        .map_instance(
            FlashInferAttention,
            layer_id=0,
            n_heads="$n_heads",
            head_dim="$head_dim",
            n_key_value_heads="$n_key_value_heads",
            apply_pos_emb_fn="$apply_pos_emb_fn",
            q_norm="$q_norm",
            k_norm="$k_norm",
        )
        .chain(),
        patch_fn.select()
        .instance(FlashInferAttention)
        .at("layer_id")
        .set_sequence("$index")
        .chain(),
    ]

    if qkv_split:
        patches += [
            patch_fn.select()
            .instance(SelfAttentionQKV)
            .map_instance(
                InferSelfAttentionQKV,
                q="$q",
                k="$k",
                v="$v",
                attention="$attention",
                out="$out",
                q_init="$q_init",
                k_init="$k_init",
                v_init="$v_init",
                out_init="$out_init",
                scaler="$scaler",
            )
            .chain()
        ]

    else:
        patches += [
            patch_fn.select()
            .instance(SelfAttention)
            .map_instance(
                InferSelfAttention,
                qkv="$qkv",
                attention="$attention",
                out="$out",
                qkv_split="llama",
                qkv_init="$qkv_init",
                out_init="$out_init",
                scaler="$scaler",
            )
            .chain()
        ]

    patches += [
        patch_fn.select()
        .instance(TransformerEncoderBlock)
        .map_instance(
            InferTransformerEncoderBlock, self_attention="$self_attention", ff="$ff"
        )
        .chain(),
        patch_fn.select()
        .instance(TransformerDecoder)
        .map_instance(
            InferTransformerDecoder,
            embedding="$embedding",
            pos_embed="$pos_embed",
            blocks="$blocks",
            post_blocks="$post_blocks",
            head="$head",
            tie_embeds="$tie_embeds",
            use_position_ids="$use_position_ids",
            attention_processor="$attention_processor",
        )
        .chain(),
    ]

    return partial(patch, patches=patches)


@config_fn
def transformer_tokainfer(
    vocab_size=None,
    dim=None,
    n_heads=None,
    head_dim=None,
    n_layers=None,
    intermediate_size=None,
    context_len=None,
    n_key_value_heads=None,
    pos_embed="rope",
    pos_embed_apply_fn=None,
    softcap=0.0,
    is_causal=True,
    norm=None,
    rms_norm_epsilon=1e-6,
    post_norm=False,
    attention_processor="auto",
    attention_bias=False,
    flex_attention_processor=None,
    qkv_split=False,
    gated_ff_split=False,
    ffn_bias=False,
    ffn_activation="swiglu",
    infer: str | None = None,
    embedding=None,
    q_norm=None,
    k_norm=None,
    use_head=True,
    tie_embeds=False,
    layer_types=None,
):
    if infer is None:
        return partial(patch, patches=())

    patches = [
        patch_fn.select()
        .instance(Attention)
        .map_instance(
            TokaInferAttention,
            layer_id=0,
            n_heads="$n_heads",
            head_dim="$head_dim",
            n_key_value_heads="$n_key_value_heads",
            apply_pos_emb_fn="$apply_pos_emb_fn",
            q_norm="$q_norm",
            k_norm="$k_norm",
        )
        .chain(),
        patch_fn.select()
        .instance(TokaInferAttention)
        .at("layer_id")
        .set_sequence("$index")
        .chain(),
    ]

    if qkv_split:
        patches += [
            patch_fn.select()
            .instance(SelfAttentionQKV)
            .map_instance(
                TokaInferSelfAttentionQKV,
                q="$q",
                k="$k",
                v="$v",
                attention="$attention",
                out="$out",
                q_init="$q_init",
                k_init="$k_init",
                v_init="$v_init",
                out_init="$out_init",
                scaler="$scaler",
            )
            .chain()
        ]

    else:
        patches += [
            patch_fn.select()
            .instance(SelfAttention)
            .map_instance(
                TokaInferSelfAttention,
                qkv="$qkv",
                attention="$attention",
                out="$out",
                qkv_split="llama",
                qkv_init="$qkv_init",
                out_init="$out_init",
                scaler="$scaler",
            )
            .chain()
        ]

    patches += [
        patch_fn.select()
        .instance(TransformerEncoderBlock)
        .map_instance(
            TokaInferTransformerEncoderBlock, self_attention="$self_attention", ff="$ff"
        )
        .chain(),
        patch_fn.select()
        .instance(TransformerDecoder)
        .map_instance(
            TokaInferTransformerDecoder,
            embedding="$embedding",
            pos_embed="$pos_embed",
            blocks="$blocks",
            post_blocks="$post_blocks",
            head="$head",
            tie_embeds="$tie_embeds",
            use_position_ids="$use_position_ids",
            attention_processor="$attention_processor",
        )
        .chain(),
    ]

    return partial(patch, patches=patches)


@config_fn
def use_flex_attention(model_conf):
    conf = select(model_conf).instance(Attention).update_dict(dict(processor="flex"))

    return conf


@config_fn
def use_flash_attention(model_conf):
    conf = (
        select(model_conf).instance(Attention).update_dict(dict(processor="flash_attn"))
    )

    return conf


@config_fn
def use_fused_linear_cross_entropy(model_conf, ignore_index=-100, z_loss=0):
    conf = (
        select(model_conf)
        .instance(VocabParallelLinear)
        .map(
            lambda x: FusedLinearCrossEntropy(
                x.linear, x.linear_init, ignore_index=ignore_index, z_loss=z_loss
            )
        )
    )

    return conf
