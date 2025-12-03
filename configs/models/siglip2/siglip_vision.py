from functools import partial

from torch import nn
from slickconf import call, field, function

from halite.transformers.parallelize import parallelize
from halite.transformers.embedding import (
    PatchEmbedding,
)
from halite.transformers.position import (
    LearnedPositionEmbedding,
)

from ..transformer import transformer
from .checkpoint_vision import weight_maps, to_halite_postprocess

conf = field()


use_complex_rope = True
qkv_split = True


conf.parallelize = partial(
    parallelize, param_dtype="bfloat16", reduce_dtype="float32", compile=True
)

conf.policy = field(
    weight_maps=weight_maps,
    to_halite_postprocess=function(to_halite_postprocess),
)


def siglip2_large_patch16_256():
    siglip_dim = 1024
    siglip_heads = 16
    siglip_head_dim = siglip_dim // siglip_heads
    n_layers = 24
    image_size = 256
    patch_size = 16
    siglip_context_len = (image_size // patch_size) ** 2
    use_conv = True

    siglip_config = field(
        vocab_size=0,
        dim=siglip_dim,
        n_heads=siglip_heads,
        head_dim=siglip_head_dim,
        n_layers=n_layers,
        context_len=siglip_context_len,
        intermediate_size=4096,
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
            use_bias=True,
            use_conv=use_conv,
        ),
        qkv_split=qkv_split,
        use_head=False,
    )

    conf.model = call[transformer](**siglip_config)
    conf.model_conf = field(
        **siglip_config,
        use_complex_rope=use_complex_rope,
        dtype="bfloat16",
        use_conv=use_conv,
    )

    return conf
