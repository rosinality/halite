from functools import partial

from slickconf import call, field, function

from halite.transformers.position import Llama3RoPE, apply_rotary_emb

from ..transformer import transformer
from .checkpoint import weight_maps, to_halite_postprocess

conf = field()

dim = 3072
n_head = 24
initial_max_position_embeddings = 8192
use_complex_rope = False
qkv_split = True

model_config = field(
    vocab_size=128256,
    dim=dim,
    n_head=n_head,
    n_layer=28,
    n_key_value_head=8,
    intermediate_size=8192,
    rms_norm_epsilon=1e-5,
    max_position_embeddings=initial_max_position_embeddings,
    pos_embed=Llama3RoPE(
        dim // n_head,
        initial_max_position_embeddings,
        use_scaled_rope=True,
        use_complex=use_complex_rope,
    ),
    pos_embed_apply_fn=partial(apply_rotary_emb, use_complex=use_complex_rope),
    qkv_split=qkv_split,
    gated_ff_split=qkv_split,
)

conf.model = call[transformer](**model_config)

conf.model_config = model_config
conf.model_config.use_complex_rope = use_complex_rope

conf.policy = field(
    weight_maps=weight_maps,
    to_halite_postprocess=function(to_halite_postprocess),
)
