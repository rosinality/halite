from slickconf import field

from ..models.hnet import hnet

conf = field()

conf.model = field()


def stage1_l():
    arch = [
        ({"mixer": "mamba2", "n_layer": 4, "use_ffn": False},),
        ({"mixer": "attention", "n_layer": 22, "use_ffn": True},),
        ({"mixer": "mamba2", "n_layer": 4, "use_ffn": False},),
    ]
    dims = [1024, 1536]
    intermediate_sizes = [0, 4096]
    pos_dims = [32, 48]
    attn_kwargs = [
        {"n_heads": 16, "head_dim": 1024 // 16, "is_causal": True},
        {"n_heads": 16, "head_dim": 1536 // 16, "is_causal": True},
    ]
    mamba2_kwargs = {"chunk_size": 256, "d_conv": 4, "d_state": 128, "expand": 2}

    conf.model.model = hnet(
        arch=arch,
        dims=dims,
        intermediate_sizes=intermediate_sizes,
        pos_dims=pos_dims,
        attn_kwargs=attn_kwargs,
        mamba2_kwargs=mamba2_kwargs,
        vocab_size=256,
        context_len=2048,
    )

    return conf
