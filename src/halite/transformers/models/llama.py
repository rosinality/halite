from functools import partial

import torch
from torch import nn

from meshfn import instantiate
from meshfn.config.builder import F, L, placeholder
from meshfn.distributed import ParallelMode
from meshfn.nn import set_meta
from meshfn.nn.activation import SwiGLU
from meshfn.nn.normalization import RMSNorm
from meshfn.nn.parallel import strategy
from meshfn.nn.parallel.tensor1d import (
    LinearColumn1D,
    LinearRow1D,
    VocabParallelEmbedding,
)
from meshfn.transformers.builder import Policy
from meshfn.transformers.builder.attention import Attention, SelfAttention
from meshfn.transformers.builder.block import TransformerEncoderBlock
from meshfn.transformers.builder.convert import get_matched_keys
from meshfn.transformers.builder.embedding import TextEmbedding
from meshfn.transformers.builder.feedforward import FeedForward, VocabParallelLinear
from meshfn.transformers.builder.normalization import PreNormalization
from meshfn.transformers.builder.position import RotaryEmbedding, make_causal_mask
from meshfn.transformers.builder.transformer import (
    TransformerConfig,
    TransformerDecoder,
)
from meshfn.utils import get_torch_dtype

weight = {"weight": "weight"}
weight_bias = {"weight": "weight", "bias": "bias"}


def llama(
    conf, parallel_context, logit=True, batch_first=True, attention_processor="native"
):
    tp = parallel_context.world_size(ParallelMode.TENSOR_1D)

    if "CausalLM" not in conf.architectures[0]:
        logit = False

    return L[llama_model](
        vocab_size=conf.vocab_size,
        dim=conf.hidden_size,
        n_head=conf.num_attention_heads,
        n_layer=conf.num_hidden_layers,
        intermediate_size=conf.intermediate_size,
        max_position_embeddings=conf.max_position_embeddings,
        rms_norm_epsilon=conf.rms_norm_eps,
        attention_dropout=0,
        hidden_dropout=0,
        tp=tp,
        logit=logit,
        batch_first=batch_first,
        attention_processor=placeholder(attention_processor),
        hf_config=conf.to_diff_dict(),
        parallel_context=placeholder(),
        dtype=placeholder(),
        device=placeholder(),
    )


def llama_model(
    vocab_size,
    dim,
    n_head,
    n_layer,
    intermediate_size,
    max_position_embeddings,
    rms_norm_epsilon,
    attention_dropout,
    hidden_dropout,
    tp,
    logit=True,
    batch_first=True,
    attention_processor="native",
    parallel_context=None,
    dtype=None,
    device=None,
    hf_config=None,
    vllm=False,
):
    dtype = get_torch_dtype(dtype)

    n_head_tp = n_head
    rms_norm = L[RMSNorm](dim, eps=rms_norm_epsilon)

    if tp > 1:
        embedding = partial(
            L[VocabParallelEmbedding], dtype=None, device=None, parallel_context=None
        )
        linear_col = partial(
            L[LinearColumn1D], dtype=None, device=None, parallel_context=None
        )
        linear_row = partial(
            L[LinearRow1D], dtype=None, device=None, parallel_context=None
        )
        linear_logit = partial(
            L[LinearColumn1D],
            gather_output=True,
            dtype=None,
            device=None,
            parallel_context=None,
        )
        n_head_tp //= tp

    else:
        embedding = L[nn.Embedding]
        linear_col = L[nn.Linear]
        linear_row = L[nn.Linear]
        linear_logit = L[nn.Linear]

    head = None

    if logit:
        head = L[VocabParallelLinear](linear_logit(dim, vocab_size, bias=False))

    transformer_config = TransformerConfig(
        dim=dim,
        n_heads=n_head,
        head_dim=dim // n_head,
        n_heads_tp=n_head_tp,
        max_length=None,
        n_layers=n_layer,
        vocab_size=vocab_size,
    )

    if attention_processor == "flash_attn":
        attention = L[Attention](
            n_head_tp,
            dim // n_head,
            attn_drop=attention_dropout,
            processor="flash_attn",
            causal=True,
        )

    else:
        attention = L[Attention](
            n_head_tp,
            dim // n_head,
            attn_drop=attention_dropout,
            processor=attention_processor,
        )

    conf = L[TransformerDecoder](
        embedding=L[TextEmbedding](
            embedding(
                vocab_size,
                dim,
            ),
            0,
        ),
        attention_mask=F[make_causal_mask](),
        pos_embed=L[RotaryEmbedding](dim // n_head, max_position_embeddings),
        blocks=[
            L[TransformerEncoderBlock](
                L[PreNormalization](
                    rms_norm,
                    L[SelfAttention](
                        qkv=linear_col(dim, dim * 3, bias=False),
                        attention=attention,
                        out=linear_row(dim, dim, bias=False),
                        qkv_split="llama",
                    ),
                    dropout=hidden_dropout,
                ),
                L[PreNormalization](
                    rms_norm,
                    L[FeedForward](
                        linear_col(dim, intermediate_size * 2, bias=False),
                        L[SwiGLU](),
                        linear_row(intermediate_size, dim, bias=False),
                    ),
                    dropout=hidden_dropout,
                ),
            )
        ]
        * n_layer,
        post_blocks=rms_norm,
        head=head,
        batch_first=batch_first,
        tie_embeds=False,
        use_position_ids=True,
        attention_processor=attention_processor,
        parallel_context=None,
    )

    model = instantiate(
        conf,
        _recursive_kwargs_=True,
        parallel_context=parallel_context,
        dtype=dtype,
        device=device,
        config=transformer_config,
    )

    if hf_config is not None:
        set_meta(model, "hf_config", hf_config)

    return model


def post_shard_process_fn(conf, state_dict):
    sd_keys = list(state_dict.keys())

    qs, qkv_keys = get_matched_keys(
        sd_keys,
        "blocks.#.self_attention.module.q.weight",
        "blocks.#.self_attention.module.qkv.weight",
    )
    ks = get_matched_keys(sd_keys, "blocks.#.self_attention.module.k.weight")
    vs = get_matched_keys(sd_keys, "blocks.#.self_attention.module.v.weight")

    linear1s, linear1_keys = get_matched_keys(
        sd_keys,
        "blocks.#.ff.module.linear1.weight",
        "blocks.#.ff.module.linear1.weight",
    )
    linear_gates = get_matched_keys(sd_keys, "blocks.#.ff.module.linear_gate.weight")

    new_sd = {}

    for q, qkv_key, k, v in zip(qs, qkv_keys, ks, vs):
        q = state_dict[q]
        k = state_dict[k]
        v = state_dict[v]

        new_sd[qkv_key] = torch.cat((q, k, v), 0)

    for linear1, linear1_key, linear_gate in zip(linear1s, linear1_keys, linear_gates):
        linear1 = state_dict[linear1]
        linear_gate = state_dict[linear_gate]

        new_sd[linear1_key] = torch.cat((linear_gate, linear1), 0)

    skip_keys = set(qs + ks + vs + linear1s + linear1_keys)

    for key, val in state_dict.items():
        if key not in skip_keys:
            new_sd[key] = val

    return new_sd


class LlamaPolicy(Policy):
    arch = llama
    weight_mappings = {
        "embed_tokens": {
            "key": "embedding.embed_input_ids",
            "map": weight,
            "strategy": strategy.VocabParallelEmbedding,
        },
        "word_embeddings_layernorm": {"key": "post_embed", "map": weight_bias},
        "layers.#.input_layernorm": {
            "key": "blocks.#.self_attention.normalization",
            "map": weight,
        },
        "layers.#.self_attn.q_proj": {
            "key": "blocks.#.self_attention.module.q",
            "map": weight,
            "strategy": strategy.LinearColumn1D,
        },
        "layers.#.self_attn.k_proj": {
            "key": "blocks.#.self_attention.module.k",
            "map": weight,
            "strategy": strategy.LinearColumn1D,
        },
        "layers.#.self_attn.v_proj": {
            "key": "blocks.#.self_attention.module.v",
            "map": weight,
            "strategy": strategy.LinearColumn1D,
        },
        "layers.#.self_attn.o_proj": {
            "key": "blocks.#.self_attention.module.out",
            "map": weight,
            "strategy": strategy.LinearRow1D,
        },
        "layers.#.post_attention_layernorm": {
            "key": "blocks.#.ff.normalization",
            "map": weight,
        },
        "layers.#.mlp.gate_proj": {
            "key": "blocks.#.ff.module.linear_gate",
            "map": weight,
            "strategy": strategy.LinearColumn1D,
        },
        "layers.#.mlp.up_proj": {
            "key": "blocks.#.ff.module.linear1",
            "map": weight,
            "strategy": strategy.LinearColumn1D,
        },
        "layers.#.mlp.down_proj": {
            "key": "blocks.#.ff.module.linear2",
            "map": weight,
            "strategy": strategy.LinearRow1D,
        },
        "norm": {"key": "post_blocks", "map": weight},
        "lm_head": {
            "key": "head.linear",
            "map": weight,
            "strategy": strategy.LinearColumn1D,
        },
    }
    post_shard_process = post_shard_process_fn
