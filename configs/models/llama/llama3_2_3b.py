from functools import partial

from slickconf import call, field, function, tag

from halite.data.tokenizers.llama3 import Llama3Tokenizer
from halite.transformers.position import Llama3RoPE, apply_rotary_emb

from ..transformer import transformer
from .checkpoint import weight_maps, to_halite_postprocess

conf = field()

dim = 3072
n_heads = 24
head_dim = dim // n_heads
context_len = 8192
use_complex_rope = True
qkv_split = True

transformer_config = field(
    vocab_size=128256,
    dim=dim,
    n_heads=n_heads,
    head_dim=head_dim,
    n_layers=28,
    n_key_value_heads=8,
    intermediate_size=8192,
    rms_norm_epsilon=1e-5,
    context_len=context_len,
    pos_embed=Llama3RoPE(
        head_dim,
        context_len,
        use_scaled_rope=True,
        use_complex=use_complex_rope,
    ),
    pos_embed_apply_fn=partial(apply_rotary_emb, use_complex=use_complex_rope),
    qkv_split=qkv_split,
    gated_ff_split=qkv_split,
)

conf.model = call[transformer](**transformer_config)
conf.model_infer = call[transformer](**transformer_config, infer="flashinfer")
conf.model_conf = field(
    **transformer_config, use_complex_rope=use_complex_rope, dtype="bfloat16"
)

conf.tokenizer = Llama3Tokenizer()

conf.policy = field(
    weight_maps=weight_maps,
    to_halite_postprocess=function(to_halite_postprocess),
)
