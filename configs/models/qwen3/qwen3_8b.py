from slickconf import call, field

from ..transformer import transformer, transformer_infer
from .checkpoint import weight_maps
from .qwen3_0_6b import conf, transformer_config

use_complex_rope = False

transformer_config.dim = 4096
transformer_config.intermediate_size = 12288
transformer_config.n_heads = 32
transformer_config.n_layers = 36
transformer_config.tie_embeds = False

conf.model = call[transformer](**transformer_config)
conf.model_infer = call[transformer_infer](**transformer_config, infer="flashinfer")
conf.model_conf = field(
    **transformer_config, use_complex_rope=use_complex_rope, dtype="bfloat16"
)

conf.policy.weight_maps = weight_maps
