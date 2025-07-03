from functools import partial

from slickconf import call, field, function, tag

from halite.data.tokenizers.hf import HFTokenizer
from halite.nn.normalization import RMSNorm
from halite.transformers.position import RotaryEmbedding, apply_rotary_emb
from halite.transformers.parallelize import parallelize

from ..transformer import transformer, transformer_infer, transformer_tokainfer
from .checkpoint import weight_maps, to_halite_postprocess

conf = field()

dim = 1024
n_heads = 16
head_dim = 128
context_len = 40960
use_complex_rope = False
qkv_split = True
rms_eps = 1e-6

transformer_config = field(
    vocab_size=151936,
    dim=dim,
    n_heads=n_heads,
    head_dim=head_dim,
    n_layers=28,
    n_key_value_heads=8,
    intermediate_size=3072,
    rms_norm_epsilon=rms_eps,
    context_len=context_len,
    pos_embed=RotaryEmbedding(
        head_dim, context_len, base=1000000, use_rotate_half=True, pre_init=False
    ),
    pos_embed_apply_fn=partial(apply_rotary_emb, use_complex=use_complex_rope),
    qkv_split=qkv_split,
    gated_ff_split=qkv_split,
    q_norm=RMSNorm(head_dim, eps=rms_eps),
    k_norm=RMSNorm(head_dim, eps=rms_eps),
    tie_embeds=True,
)

conf.model = call[transformer](**transformer_config)
conf.model_infer = call[transformer_infer](**transformer_config, infer="flashinfer")
conf.model_conf = field(
    **transformer_config, use_complex_rope=use_complex_rope, dtype="bfloat16"
)

conf.parallelize = partial(
    parallelize, param_dtype="bfloat16", reduce_dtype="float32", compile=True
)

conf.tokenizer = HFTokenizer()

weight_maps = {**weight_maps}
del weight_maps["lm_head"]

conf.policy = field(
    weight_maps=weight_maps,
    to_halite_postprocess=function(to_halite_postprocess),
)
