from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    cascade,
)
import torch
from torch import Tensor

from halite.transformers.tokainfer.types import (
    AttentionInfo,
    DeviceType,
    WrapperCollection,
)


def create_workspace_buffer(device: DeviceType):
    # flashinfer recommends a 128MB buffer
    return torch.empty(
        128 * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )


def create_wrappers(
    device: DeviceType,
    num_attention_heads: int,
    num_key_value_heads: int,
    workspace_buffer: Tensor | None = None,
):
    if workspace_buffer is None:
        workspace_buffer = create_workspace_buffer(device)

    gqa_ratio = num_attention_heads // num_key_value_heads

    # NOTE: I think it's ok to reuse the buffers across both wrappers
    prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)
    hydragen_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)
    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, use_tensor_cores=gqa_ratio >= 4
    )

    return WrapperCollection(
        prefill_wrapper=prefill_wrapper,
        hydragen_wrapper=hydragen_wrapper,
        decode_wrapper=decode_wrapper,
    )


def create_wrappers_for_cudagraph(
    device: DeviceType,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_decode_sequences: int,
    max_kv_indices: int,
    workspace_buffer: Tensor | None = None,
):
    if workspace_buffer is None:
        workspace_buffer = create_workspace_buffer(device)

    gqa_ratio = num_attention_heads // num_key_value_heads

    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        use_tensor_cores=gqa_ratio >= 4,
        use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.empty(
            num_decode_sequences + 1,
            dtype=torch.int32,
            device=device,
        ),
        paged_kv_indices_buffer=torch.empty(
            max_kv_indices, dtype=torch.int32, device=device
        ),
        paged_kv_last_page_len_buffer=torch.empty(
            num_decode_sequences, dtype=torch.int32, device=device
        ),
    )

    return WrapperCollection(
        prefill_wrapper=None,
        hydragen_wrapper=None,
        decode_wrapper=decode_wrapper,
    )


def append_to_kv_cache(
    token_indices: Tensor,
    key: Tensor,
    value: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
):
    """
    Important to back out of torch compile for this op, since the compiler
    seemed to be making a copy of the cache, taking a lot of mem/time.
    """

    _, num_key_value_heads, head_dim = key.shape

    flat_k_cache = k_cache.view(-1, num_key_value_heads, head_dim)
    flat_v_cache = v_cache.view(-1, num_key_value_heads, head_dim)

    flat_k_cache[token_indices] = key
    flat_v_cache[token_indices] = value


def tokainfer_attention(
    ragged_q: Tensor,
    ragged_k: Tensor,
    ragged_v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    attn_info: AttentionInfo,
    wrappers: WrapperCollection,
) -> Tensor:
    """
    Assumes rope has been already applied.
    """

    append_to_kv_cache(
        token_indices=attn_info.append_kv_token_indices,
        key=ragged_k,
        value=ragged_v,
        k_cache=k_cache,
        v_cache=v_cache,
    )

    prefill_q, hydragen_q, decode_q = attn_info.split_q(ragged_q)

    # the key difference between the hydragen shared
    # prefix attention and normal prefill
    # is that hydragen does not have a causal mask
    if prefill_q.numel() > 0:
        prefill_wrapper = wrappers.prefill_wrapper
        assert prefill_wrapper is not None
        true_prefill_output = prefill_wrapper.run(
            q=prefill_q, paged_kv_cache=(k_cache, v_cache)
        )
    else:
        true_prefill_output = prefill_q

    # decode
    if decode_q.numel() > 0:
        decode_wrapper = wrappers.decode_wrapper
        assert decode_wrapper is not None
        decode_output, decode_lse = decode_wrapper.run_return_lse(
            q=decode_q, paged_kv_cache=(k_cache, v_cache)
        )
    else:
        decode_output = decode_q

    if hydragen_q.numel() > 0:
        hydragen_wrapper = wrappers.hydragen_wrapper
        assert hydragen_wrapper is not None
        shared_prefill_output, shared_prefill_lse = hydragen_wrapper.run_return_lse(
            q=hydragen_q, paged_kv_cache=(k_cache, v_cache)
        )

        # Unique (decode)
        n_mixed = attn_info.hydragen_info.num_tokens
        unique_lse = decode_lse[:n_mixed]
        unique_out = decode_output[:n_mixed]

        aggregate, _ = cascade.merge_state(
            shared_prefill_output, shared_prefill_lse, unique_out, unique_lse
        )

        true_decode_out = decode_output[n_mixed:]
        output = torch.cat([true_prefill_output, aggregate, true_decode_out], dim=0)

    else:
        output = torch.cat([true_prefill_output, decode_output], dim=0)

    return output
