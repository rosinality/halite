from functools import partial

from torch import nn
from slickconf import call, field, function

from halite.transformers.container import MultiModule
from halite.transformers.parallelize import parallelize
from halite.transformers.embedding import (
    PatchEmbedding,
    MultiModalEmbedding,
)
from halite.transformers.flex_attention import FlexAttentionProcessor
from halite.transformers.flex_attention.bidirectional import BidirectionalMask
from halite.transformers.flex_attention.causal import CausalMask
from halite.transformers.flex_attention.sliding_window import SlidingWindowCausalMask
from halite.transformers.position import (
    LearnedPositionEmbedding,
    LinearRoPE,
    RotaryEmbedding,
    apply_rotary_emb,
)
from halite.transformers.variants.gemma3 import (
    TextEmbedding,
    RMSNorm,
    MultiModalProjector,
)

from ..transformer import transformer, transformer_infer
from .checkpoint import weight_maps, to_halite_postprocess

conf = field()

siglip_dim = 1152
siglip_heads = 16
siglip_head_dim = siglip_dim // siglip_heads
siglip_context_len = 8192
use_complex_rope = True
qkv_split = True
image_size = 896
patch_size = 14
siglip_context_len = (image_size // patch_size) ** 2

siglip_config = field(
    vocab_size=0,
    dim=siglip_dim,
    n_heads=siglip_heads,
    head_dim=siglip_head_dim,
    n_layers=27,
    context_len=siglip_context_len,
    intermediate_size=4304,
    norm=nn.LayerNorm(siglip_dim, eps=1e-6),
    pos_embed=None,
    attention_bias=True,
    is_causal=False,
    ffn_bias=True,
    ffn_activation="gelu_tanh",
    embedding=PatchEmbedding(
        3,
        siglip_dim,
        patch_size,
        LearnedPositionEmbedding(siglip_dim, siglip_context_len),
    ),
    qkv_split=qkv_split,
    use_head=False,
)

siglip = call[transformer](**siglip_config)

n_vocab = 262208
dim = 2560
n_layers = 34
n_heads = 8
head_dim = 256
context_len = 131072
image_token_id = 262144
sliding_window_ratio = 6

layer_types = [
    "local_attn" if (id + 1) % sliding_window_ratio != 0 else "global_attn"
    for id in range(n_layers)
]

decoder_config = field(
    vocab_size=n_vocab,
    dim=dim,
    n_heads=n_heads,
    head_dim=head_dim,
    n_layers=n_layers,
    context_len=siglip_context_len,
    intermediate_size=10240,
    norm=RMSNorm(dim, eps=1e-6),
    attention_bias=False,
    attention_processor="flex",
    n_key_value_heads=4,
    flex_attention_processor=MultiModule(
        local_attn=FlexAttentionProcessor(
            block_mask=BidirectionalMask(SlidingWindowCausalMask(1024))
        ),
        global_attn=FlexAttentionProcessor(block_mask=BidirectionalMask(CausalMask())),
    ),
    is_causal=False,
    ffn_bias=False,
    ffn_activation="geglu_tanh",
    pos_embed=MultiModule(
        local_attn=RotaryEmbedding(
            head_dim, context_len, base=10000.0, use_rotate_half=True
        ),
        global_attn=LinearRoPE(
            head_dim, context_len, base=1000000.0, scaling=8.0, use_rotate_half=True
        ),
    ),
    pos_embed_apply_fn=partial(apply_rotary_emb),
    post_norm=True,
    embedding=MultiModalEmbedding(
        text_embedding=TextEmbedding(nn.Embedding(n_vocab, dim), multiplier=dim**0.5),
        image_embedding=MultiModalProjector(
            siglip, siglip_dim, dim, 1e-6, image_size, patch_size, 256
        ),
        image_mask_token_id=image_token_id,
    ),
    qkv_split=qkv_split,
    gated_ff_split=True,
    q_norm=RMSNorm(head_dim, eps=1e-6),
    k_norm=RMSNorm(head_dim, eps=1e-6),
    tie_embeds=True,
    layer_types=layer_types,
)

conf.model = call[transformer](**decoder_config)
conf.model_conf = field(
    **decoder_config, use_complex_rope=use_complex_rope, dtype="bfloat16"
)

conf.parallelize = partial(
    parallelize, param_dtype="bfloat16", reduce_dtype="float32", compile=True
)

conf.policy = field(
    weight_maps=weight_maps,
    to_halite_postprocess=function(to_halite_postprocess),
)
