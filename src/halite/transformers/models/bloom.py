from functools import partial

from torch import nn

from meshfn import instantiate
from meshfn.config.builder import F, L, placeholder
from meshfn.distributed import ParallelMode
from meshfn.nn import set_meta
from meshfn.nn.activation import BloomGELU
from meshfn.nn.parallel import strategy
from meshfn.nn.parallel.tensor1d import (
    LinearColumn1D,
    LinearRow1D,
    VocabParallelEmbedding,
)
from meshfn.transformers.builder import Policy
from meshfn.transformers.builder.attention import Attention, SelfAttention
from meshfn.transformers.builder.block import (
    FusedTransformerEncoderBlock,
    TransformerEncoderBlock,
)
from meshfn.transformers.builder.convert import get_matched_keys
from meshfn.transformers.builder.embedding import TextEmbedding
from meshfn.transformers.builder.feedforward import FeedForward, VocabParallelLinear
from meshfn.transformers.builder.normalization import (
    FusedPostBlock,
    FusedPreNormalization,
    PreNormalization,
)
from meshfn.transformers.builder.position import ALiBi, alibi_params, make_causal_mask
from meshfn.transformers.builder.schema import plain_transformer_encoder_schema
from meshfn.transformers.builder.transformer import (
    TransformerConfig,
    TransformerDecoder,
)
from meshfn.transformers.builder.vllm.attention import (
    PagedAttentionWithALiBi,
    vLLMSelfAttention,
)
from meshfn.transformers.builder.vllm.block import vLLMTransformerEncoderBlock
from meshfn.transformers.builder.vllm.transformer import vLLMTransformerDecoder
from meshfn.utils import get_torch_dtype

weight = {"weight": "weight"}
weight_bias = {"weight": "weight", "bias": "bias"}


def permute_attention_weight(conf, weight):
    out_dim, in_dim = weight.shape
    weight = weight.reshape(conf.n_head, 3, out_dim // 3 // conf.n_head, in_dim)
    weight = weight.transpose(1, 2)
    weight = weight.reshape(out_dim, in_dim)

    return weight


def permute_attention_bias(conf, bias):
    out_dim = bias.shape[0]
    bias = bias.reshape(conf.n_head, 3, out_dim // 3 // conf.n_head)
    bias = bias.transpose(1, 2)
    bias = bias.reshape(out_dim)

    return bias


def bloom(
    conf,
    parallel_context,
    logit=True,
    batch_first=True,
    attention_processor="native",
):
    tp = parallel_context.world_size(ParallelMode.TENSOR_1D)

    logit = any("CausalLM" in arch for arch in conf.architectures)

    return L[bloom_model](
        vocab_size=conf.vocab_size,
        dim=conf.hidden_size,
        n_head=conf.n_head,
        n_layer=conf.n_layer,
        layer_norm_epsilon=conf.layer_norm_epsilon,
        attention_dropout=conf.attention_dropout,
        hidden_dropout=conf.hidden_dropout,
        tp=tp,
        logit=logit,
        batch_first=batch_first,
        attention_processor=placeholder(attention_processor),
        fused_norm=placeholder(False),
        hf_config=conf.to_diff_dict(),
        parallel_context=placeholder(),
        dtype=placeholder(),
        device=placeholder(),
    )


def bloom_model(
    vocab_size,
    dim,
    n_head,
    n_layer,
    layer_norm_epsilon,
    attention_dropout,
    hidden_dropout,
    tp,
    logit=True,
    batch_first=True,
    attention_processor="native",
    fused_norm=False,
    parallel_context=None,
    dtype=None,
    device=None,
    hf_config=None,
    vllm=False,
):
    dtype = get_torch_dtype(dtype)

    n_head_tp = n_head
    layer_norm = L[nn.LayerNorm](dim, eps=layer_norm_epsilon)

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

    unpad = False
    post_blocks = layer_norm
    normalization_block = PreNormalization
    if vllm:
        transformer_decoder = vLLMTransformerDecoder
        transformer_block = vLLMTransformerEncoderBlock
        attention = L[PagedAttentionWithALiBi](
            n_head_tp,
            dim // n_head,
        )
        self_attention = vLLMSelfAttention

    else:
        transformer_decoder = TransformerDecoder
        transformer_block = TransformerEncoderBlock

        if attention_processor == "flash_attn":
            alibi_start, alibi_ratio = alibi_params(
                n_head, parallel_context.local_rank(ParallelMode.TENSOR_1D), tp
            )

            attention = L[Attention](
                n_head_tp,
                dim // n_head,
                attn_drop=attention_dropout,
                processor="flash_attn",
                causal=True,
                alibi=True,
                alibi_start=alibi_start,
                alibi_ratio=alibi_ratio,
            )
            unpad = True

        else:
            attention = L[Attention](
                n_head_tp,
                dim // n_head,
                attn_drop=attention_dropout,
                processor=attention_processor,
            )

        self_attention = SelfAttention

    if not vllm and fused_norm:
        transformer_block = FusedTransformerEncoderBlock
        normalization_block = FusedPreNormalization
        post_blocks = L[FusedPostBlock](dim, eps=layer_norm_epsilon)

    conf = L[transformer_decoder](
        embedding=L[TextEmbedding](
            embedding(
                vocab_size,
                dim,
            ),
            0,
        ),
        post_embed=layer_norm,
        attention_mask=F[make_causal_mask](),
        pos_embed=L[ALiBi](n_head),
        blocks=[
            L[transformer_block](
                L[normalization_block](
                    layer_norm,
                    L[self_attention](
                        qkv=linear_col(dim, dim * 3),
                        attention=attention,
                        out=linear_row(dim, dim),
                    ),
                    dropout=hidden_dropout,
                ),
                L[normalization_block](
                    layer_norm,
                    L[FeedForward](
                        linear_col(dim, dim * 4),
                        # using nn.GELU vs BloomGelu can make differences,
                        # especially in low precisions.
                        L[BloomGELU](),
                        # L[nn.GELU](approximate="tanh"),
                        linear_row(dim * 4, dim),
                    ),
                    dropout=hidden_dropout,
                ),
            )
        ]
        * n_layer,
        post_blocks=post_blocks,
        head=head,
        batch_first=batch_first,
        tie_embeds=True,
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


def pre_unshard_process_fn(conf, state_dict):
    sd_keys = list(state_dict.keys())

    qkvs, qkv_keys = get_matched_keys(
        sd_keys,
        "blocks.#.self_attention.module.qkv.weight",
        "blocks.#.self_attention.module.qkv.weight",
    )
    qkv_biases, qkv_biases_keys = get_matched_keys(
        sd_keys,
        "blocks.#.self_attention.module.qkv.bias",
        "blocks.#.self_attention.module.qkv.bias",
    )

    new_sd = {}
    head_dim = conf["hidden_size"] // conf["n_head"]

    for qkv, qkv_key in zip(qkvs, qkv_keys):
        qkv = state_dict[qkv]

        out_dim, in_dim = qkv.shape
        new_sd[qkv_key] = (
            qkv.reshape(3, -1, head_dim, in_dim)
            .transpose(0, 1)
            .reshape(out_dim, in_dim)
        )

    for qkv_bias, qkv_bias_key in zip(qkv_biases, qkv_biases_keys):
        qkv_bias = state_dict[qkv_bias]

        out_dim = qkv_bias.shape[0]
        new_sd[qkv_bias_key] = (
            qkv_bias.reshape(3, -1, head_dim).transpose(0, 1).reshape(-1)
        )

    skip_keys = set(qkv_keys + qkv_biases_keys)

    for key, val in state_dict.items():
        if key not in skip_keys:
            new_sd[key] = val

    return new_sd


def post_shard_process_fn(conf, state_dict):
    sd_keys = list(state_dict.keys())

    qkvs, qkv_keys = get_matched_keys(
        sd_keys,
        "blocks.#.self_attention.module.qkv.weight",
        "blocks.#.self_attention.module.qkv.weight",
    )
    qkv_biases, qkv_biases_keys = get_matched_keys(
        sd_keys,
        "blocks.#.self_attention.module.qkv.bias",
        "blocks.#.self_attention.module.qkv.bias",
    )

    new_sd = {}

    head_dim = conf.hidden_size // conf.n_head

    for qkv, qkv_key in zip(qkvs, qkv_keys):
        qkv = state_dict[qkv]

        out_dim, in_dim = qkv.shape
        new_sd[qkv_key] = (
            qkv.reshape(-1, 3, head_dim, in_dim)
            .transpose(0, 1)
            .reshape(out_dim, in_dim)
        )

    for qkv_bias, qkv_bias_key in zip(qkv_biases, qkv_biases_keys):
        qkv_bias = state_dict[qkv_bias]

        out_dim = qkv_bias.shape[0]
        new_sd[qkv_bias_key] = (
            qkv_bias.reshape(-1, 3, head_dim).transpose(0, 1).reshape(-1)
        )

    skip_keys = set(qkv_keys + qkv_biases_keys)

    for key, val in state_dict.items():
        if key not in skip_keys:
            new_sd[key] = val

    return new_sd


class BloomPolicy(Policy):
    arch = bloom
    schema = plain_transformer_encoder_schema
    weight_mappings = {
        "word_embeddings": {
            "key": "embedding.embed_input_ids",
            "map": weight,
            "strategy": strategy.VocabParallelEmbedding,
        },
        "word_embeddings_layernorm": {"key": "post_embed", "map": weight_bias},
        "h.#.input_layernorm": {
            "key": "blocks.#.self_attention.normalization",
            "map": weight_bias,
        },
        "h.#.self_attention.query_key_value": {
            "key": "blocks.#.self_attention.module.qkv",
            "map": weight_bias,
            "strategy": strategy.LinearColumn1D,
        },
        "h.#.self_attention.dense": {
            "key": "blocks.#.self_attention.module.out",
            "map": weight_bias,
            "strategy": strategy.LinearRow1D,
        },
        "h.#.post_attention_layernorm": {
            "key": "blocks.#.ff.normalization",
            "map": weight_bias,
        },
        "h.#.mlp.dense_h_to_4h": {
            "key": "blocks.#.ff.module.linear1",
            "map": weight_bias,
            "strategy": strategy.LinearColumn1D,
        },
        "h.#.mlp.dense_4h_to_h": {
            "key": "blocks.#.ff.module.linear2",
            "map": weight_bias,
            "strategy": strategy.LinearRow1D,
        },
        "ln_f": {"key": "post_blocks", "map": weight_bias},
        "lm_head": {
            "key": "head.linear",
            "map": weight,
            "strategy": strategy.LinearColumn1D,
        },
    }

    prefixes = {"BloomForCausalLM": {"*": "transformer", "lm_head": None}}

    pre_unshard_process = pre_unshard_process_fn
    post_shard_process = post_shard_process_fn
