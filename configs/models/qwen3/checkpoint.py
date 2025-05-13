import torch

from halite.transformers.convert import get_matched_keys


def to_halite_postprocess(config, state_dict):
    sd_keys = list(state_dict.keys())

    if config.qkv_split:
        qs = get_matched_keys(
            sd_keys,
            "blocks.#.self_attention.module.q.weight",
        )
        qkv_keys = qs

    else:
        qs, qkv_keys = get_matched_keys(
            sd_keys,
            "blocks.#.self_attention.module.q.weight",
            "blocks.#.self_attention.module.qkv.weight",
        )

    ks = get_matched_keys(sd_keys, "blocks.#.self_attention.module.k.weight")
    vs = get_matched_keys(sd_keys, "blocks.#.self_attention.module.v.weight")

    new_sd = {}

    for q_name, qkv_key, k_name, v_name in zip(qs, qkv_keys, ks, vs):
        q = state_dict[q_name]
        k = state_dict[k_name]
        v = state_dict[v_name]

        if config.qkv_split:
            new_sd[q_name] = q
            new_sd[k_name] = k
            new_sd[v_name] = v

        else:
            new_sd[qkv_key] = torch.cat((q, k, v), 0)

    if config.gated_ff_split:
        linear1s, linear1_keys = get_matched_keys(
            sd_keys,
            "blocks.#.ff.module.linear1.weight",
            "blocks.#.ff.module.linear_proj.weight",
        )
        linear_gates = get_matched_keys(
            sd_keys, "blocks.#.ff.module.linear_gate.weight"
        )
        linear2s, linear2_keys = get_matched_keys(
            sd_keys,
            "blocks.#.ff.module.linear2.weight",
            "blocks.#.ff.module.linear_out.weight",
        )

        for linear1, linear1_key, linear2, linear2_key in zip(
            linear1s, linear1_keys, linear2s, linear2_keys
        ):
            linear1 = state_dict[linear1]
            linear2 = state_dict[linear2]

            new_sd[linear1_key] = linear1
            new_sd[linear2_key] = linear2

        skip_keys = set(qs + ks + vs + linear1s + linear2s)

    else:
        linear1s, linear1_keys = get_matched_keys(
            sd_keys,
            "blocks.#.ff.module.linear1.weight",
            "blocks.#.ff.module.linear1.weight",
        )
        linear_gates = get_matched_keys(
            sd_keys, "blocks.#.ff.module.linear_gate.weight"
        )

        for linear1, linear1_key, linear_gate in zip(
            linear1s, linear1_keys, linear_gates
        ):
            linear1 = state_dict[linear1]
            linear_gate = state_dict[linear_gate]

            new_sd[linear1_key] = torch.cat((linear_gate, linear1), 0)

        skip_keys = set(qs + ks + vs + linear1s + linear_gates)

    for key, val in state_dict.items():
        if key not in skip_keys:
            new_sd[key] = val

    return new_sd


weight_maps = {
    "model.embed_tokens": {
        "key": "embedding.embed_input_ids",
        "placement": {
            "weight": ("shard", 0),
        },
    },
    "model.layers.#.input_layernorm": {
        "key": "blocks.#.self_attention.pre_norm",
        "placement": {"weight": "replicate"},
    },
    "model.layers.#.self_attn.q_proj": {
        "key": "blocks.#.self_attention.module.q",
        "placement": {
            "weight": ("shard", 0),
        },
    },
    "model.layers.#.self_attn.k_proj": {
        "key": "blocks.#.self_attention.module.k",
        "placement": {
            "weight": ("shard", 0),
        },
    },
    "model.layers.#.self_attn.v_proj": {
        "key": "blocks.#.self_attention.module.v",
        "placement": {
            "weight": ("shard", 0),
        },
    },
    "model.layers.#.self_attn.o_proj": {
        "key": "blocks.#.self_attention.module.out",
        "placement": {
            "weight": ("shard", 1),
        },
    },
    "model.layers.#.self_attn.q_norm": {
        "key": "blocks.#.self_attention.module.attention.q_norm",
        "placement": {"weight": "replicate"},
    },
    "model.layers.#.self_attn.k_norm": {
        "key": "blocks.#.self_attention.module.attention.k_norm",
        "placement": {"weight": "replicate"},
    },
    "model.layers.#.post_attention_layernorm": {
        "key": "blocks.#.ff.pre_norm",
        "placement": {"weight": "replicate"},
    },
    "model.layers.#.mlp.gate_proj": {
        "key": "blocks.#.ff.module.linear_gate",
        "placement": {
            "weight": ("shard", 0),
        },
    },
    "model.layers.#.mlp.down_proj": {
        "key": "blocks.#.ff.module.linear2",
        "placement": {
            "weight": ("shard", 1),
        },
    },
    "model.layers.#.mlp.up_proj": {
        "key": "blocks.#.ff.module.linear1",
        "placement": {
            "weight": ("shard", 0),
        },
    },
    "model.norm": {"key": "post_blocks.module", "placement": {"weight": "replicate"}},
    "lm_head": {
        "key": "head.linear",
        "placement": {
            "weight": ("shard", 0),
        },
    },
}
