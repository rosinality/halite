from halite.transformers.convert import get_matched_keys

siglip_prefix = "embedding.image_embedding.encoder."


def to_halite_postprocess(config, state_dict):
    sd_keys = list(state_dict.keys())

    patch_embeddings = get_matched_keys(
        sd_keys, siglip_prefix + "embedding.patch_embedding.weight"
    )
    for key in patch_embeddings:
        conv_weight = state_dict[key]
        state_dict[key] = conv_weight.reshape(conv_weight.shape[0], -1)

    return state_dict


weight_maps = {
    "vision_model.embeddings.patch_embedding": {
        "key": siglip_prefix + "embedding.patch_embedding",
        "placement": {
            "weight": "replicate",
            "bias": "replicate",
        },
    },
    "vision_model.embeddings.position_embedding": {
        "key": siglip_prefix + "embedding.position_embedding",
        "name_map": {
            "weight": "pos_embed",
        },
        "placement": {
            "pos_embed": "replicate",
        },
    },
    "vision_model.encoder.layers.#.layer_norm1": {
        "key": siglip_prefix + "blocks.#.self_attention.pre_norm",
        "placement": {"weight": "replicate", "bias": "replicate"},
    },
    "vision_model.encoder.layers.#.layer_norm2": {
        "key": siglip_prefix + "blocks.#.ff.pre_norm",
        "placement": {"weight": "replicate", "bias": "replicate"},
    },
    "vision_model.encoder.layers.#.self_attn.q_proj": {
        "key": siglip_prefix + "blocks.#.self_attention.module.q",
        "placement": {
            "weight": "replicate",
            "bias": "replicate",
        },
    },
    "vision_model.encoder.layers.#.self_attn.k_proj": {
        "key": siglip_prefix + "blocks.#.self_attention.module.k",
        "placement": {
            "weight": "replicate",
            "bias": "replicate",
        },
    },
    "vision_model.encoder.layers.#.self_attn.v_proj": {
        "key": siglip_prefix + "blocks.#.self_attention.module.v",
        "placement": {
            "weight": "replicate",
            "bias": "replicate",
        },
    },
    "vision_model.encoder.layers.#.self_attn.out_proj": {
        "key": siglip_prefix + "blocks.#.self_attention.module.out",
        "placement": {
            "weight": "replicate",
            "bias": "replicate",
        },
    },
    "vision_model.encoder.layers.#.mlp.fc1": {
        "key": siglip_prefix + "blocks.#.ff.module.linear1",
        "placement": {
            "weight": "replicate",
            "bias": "replicate",
        },
    },
    "vision_model.encoder.layers.#.mlp.fc2": {
        "key": siglip_prefix + "blocks.#.ff.module.linear2",
        "placement": {
            "weight": "replicate",
            "bias": "replicate",
        },
    },
    "vision_model.post_layernorm": {
        "key": siglip_prefix + "post_blocks.module",
        "placement": {"weight": "replicate", "bias": "replicate"},
    },
    "multi_modal_projector.mm_soft_emb_norm": {
        "key": "embedding.image_embedding.soft_embed_norm",
        "placement": {"weight": "replicate"},
    },
    "multi_modal_projector": {
        "key": "embedding.image_embedding",
        "name_map": {
            "mm_input_projection_weight": "input_proj",
        },
        "placement": {"input_proj": "replicate"},
    },
    "language_model.model.embed_tokens": {
        "key": "embedding.text_embedding.embed_input_ids",
        "placement": {
            "weight": "replicate",
        },
    },
    "language_model.model.layers.#.input_layernorm": {
        "key": "blocks.#.self_attention.pre_norm",
        "placement": {"weight": "replicate"},
    },
    "language_model.model.layers.#.self_attn.q_proj": {
        "key": "blocks.#.self_attention.module.q",
        "placement": {
            "weight": "replicate",
        },
    },
    "language_model.model.layers.#.self_attn.k_proj": {
        "key": "blocks.#.self_attention.module.k",
        "placement": {
            "weight": "replicate",
        },
    },
    "language_model.model.layers.#.self_attn.v_proj": {
        "key": "blocks.#.self_attention.module.v",
        "placement": {"weight": "replicate"},
    },
    "language_model.model.layers.#.self_attn.o_proj": {
        "key": "blocks.#.self_attention.module.out",
        "placement": {"weight": "replicate"},
    },
    "language_model.model.layers.#.self_attn.q_norm": {
        "key": "blocks.#.self_attention.module.attention.q_norm",
        "placement": {"weight": "replicate"},
    },
    "language_model.model.layers.#.self_attn.k_norm": {
        "key": "blocks.#.self_attention.module.attention.k_norm",
        "placement": {"weight": "replicate"},
    },
    "language_model.model.layers.#.post_attention_layernorm": {
        "key": "blocks.#.self_attention.post_norm",
        "placement": {"weight": "replicate"},
    },
    "language_model.model.layers.#.pre_feedforward_layernorm": {
        "key": "blocks.#.ff.pre_norm",
        "placement": {"weight": "replicate"},
    },
    "language_model.model.layers.#.mlp.gate_proj": {
        "key": "blocks.#.ff.module.linear_gate",
        "placement": {
            "weight": "replicate",
        },
    },
    "language_model.model.layers.#.mlp.up_proj": {
        "key": "blocks.#.ff.module.linear_proj",
        "placement": {
            "weight": "replicate",
        },
    },
    "language_model.model.layers.#.mlp.down_proj": {
        "key": "blocks.#.ff.module.linear_out",
        "placement": {"weight": "replicate"},
    },
    "language_model.model.layers.#.post_feedforward_layernorm": {
        "key": "blocks.#.ff.post_norm",
        "placement": {"weight": "replicate"},
    },
    "language_model.model.norm": {
        "key": "post_blocks.module",
        "placement": {"weight": "replicate"},
    },
}
