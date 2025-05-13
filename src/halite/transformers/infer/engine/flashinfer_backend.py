from regex import B
import torch
from torch import nn
import triton
import triton.language as tl
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchDecodeWithPagedKVCacheWrapper,
)

from halite.transformers.infer.engine.batch import Batch, ForwardMode
from halite.transformers.infer.engine.memory_pool import RequestToTokenPool


@triton.jit
def calc_kv_indices_kernel(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    max_context_len: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    req_to_token_ptr += req_pool_index * max_context_len
    kv_indices_ptr += kv_indices_offset

    ld_offset = kv_start + tl.arange(0, BLOCK_SIZE)
    st_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = ld_offset < kv_end
        data = tl.load(req_to_token_ptr + ld_offset, mask=mask)
        tl.store(kv_indices_ptr + st_offset, data, mask=mask)
        ld_offset += BLOCK_SIZE
        st_offset += BLOCK_SIZE


def should_use_tensor_core(kv_cache_dtype, n_qo_heads, n_kv_heads):
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        if not _grouped_size_compiled_for_decode_kernels(n_qo_heads, n_kv_heads):
            return True

        else:
            return False

    except (ImportError, AttributeError):
        pass

    gqa_group_size = n_qo_heads // n_kv_heads

    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True

    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size > 4

    else:
        return False


class FlashInferBackend:
    def __init__(
        self,
        n_qo_heads: int,
        n_kv_heads: int,
        head_dim: int,
        is_causal: bool,
        scale: float,
        dtype: torch.dtype,
        request_to_token: RequestToTokenPool,
        max_batch_size: int,
        max_context_len: int,
        device: str = "cuda",
        workspace_size: int = 128 * 1024 * 1024,
    ):
        self.use_tensor_cores = should_use_tensor_core(dtype, n_qo_heads, n_kv_heads)

        self.n_qo_heads = n_qo_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.scale = scale
        self.request_to_token = request_to_token
        self.device = device
        self.dtype = dtype
        self.workspace_size = workspace_size

        self.max_batch_size = max_batch_size
        self.max_context_len = max_context_len

        self.initialized = False
        self.initialize()

    def initialize(self):
        if self.initialized:
            return

        self.kv_indptr = torch.zeros(
            self.max_batch_size + 1, dtype=torch.int32, device=self.device
        )
        self.qo_indptr = torch.zeros(
            self.max_batch_size + 1, dtype=torch.int32, device=self.device
        )
        self.kv_last_page_len = torch.ones(
            self.max_batch_size, dtype=torch.int32, device=self.device
        )

        self.prefill = BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(self.workspace_size, dtype=torch.uint8, device=self.device),
            "NHD",
        )

        self.decode = BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(self.workspace_size, dtype=torch.uint8, device=self.device),
            "NHD",
            use_tensor_cores=self.use_tensor_cores,
        )

        self.initialized = True

    def cleanup(self):
        del self.kv_indptr
        del self.qo_indptr
        del self.kv_last_page_len
        del self.prefill
        del self.decode

        self.initialized = False

    def get_wrapper(self):
        return self.prefill, self.decode

    def prepare(self, batch):
        if batch.mode == ForwardMode.DECODE:
            self.decode_update(
                batch.request_pool_ids, batch.seq_lens, batch.seq_lens_sum
            )

        else:
            self.prefill_update(
                batch.request_pool_ids, batch.seq_lens, batch.prefix_lens
            )

    def prefill_update(self, request_to_pool_id, seq_lens, extend_prefix_lens):
        prefill, _ = self.get_wrapper()

        batch = len(request_to_pool_id)

        kv_indptr = self.kv_indptr
        kv_indptr[1 : batch + 1] = torch.cumsum(seq_lens, dim=0)
        kv_indptr = kv_indptr[: batch + 1]

        kv_indices = torch.empty(kv_indptr[-1], dtype=torch.int32, device=self.device)
        calc_kv_indices_kernel[(batch,)](
            self.request_to_token.request_to_token,
            request_to_pool_id,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.request_to_token.max_context_len,
        )

        qo_indptr = self.qo_indptr
        qo_indptr[1 : batch + 1] = torch.cumsum(seq_lens - extend_prefix_lens, dim=0)
        qo_indptr = qo_indptr[: batch + 1]

        prefill.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:batch],
            self.n_qo_heads,
            self.n_kv_heads,
            self.head_dim,
            page_size=1,
            causal=self.is_causal,
            sm_scale=self.scale,
            q_data_type=self.dtype,
        )

    def decode_update(self, request_to_pool_id, seq_lens, seq_lens_sum):
        _, decode = self.get_wrapper()

        batch = len(request_to_pool_id)

        kv_indptr = self.kv_indptr
        kv_indptr[1 : batch + 1] = torch.cumsum(seq_lens, dim=0)
        kv_indptr = kv_indptr[: batch + 1]

        kv_indices = torch.empty(seq_lens_sum, dtype=torch.int32, device=self.device)
        calc_kv_indices_kernel[(batch,)](
            self.request_to_token.request_to_token,
            request_to_pool_id,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.request_to_token.max_context_len,
        )

        decode.plan(
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:batch],
            self.n_qo_heads,
            self.n_kv_heads,
            self.head_dim,
            page_size=1,
            q_data_type=self.dtype,
            data_type=self.dtype,
        )


class FlashInferAttention(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()

        self.layer_id = layer_id

    def forward(self, q, k, v, batch: Batch):
        k = k.view(-1, k.shape[-2], v.shape[-1])
        v = v.view(-1, v.shape[-2], v.shape[-1])

        if batch.mode == ForwardMode.EXTEND:
            return self.forward_extend(q, k, v, batch)

        elif batch.mode == ForwardMode.DECODE:
            return self.forward_decode(q, k, v, batch)

    def forward_extend(self, q, k, v, batch):
        prefill, _ = batch.attention_backend.get_wrapper()
        kv_pool_ids = batch.kv_pool_ids

        if k is not None:
            batch.kv_pool.set_kv_buffer(self.layer_id, kv_pool_ids, k, v)

        kb, vb = batch.kv_pool.get_kv_buffer(self.layer_id)

        o = prefill.run(
            q.contiguous().view(
                -1, batch.attention_backend.n_qo_heads, batch.attention_backend.head_dim
            ),
            batch.kv_pool.get_kv_buffer(self.layer_id),
        )

        return o.view(
            -1,
            batch.attention_backend.n_qo_heads * batch.attention_backend.head_dim,
        )

    def forward_decode(self, q, k, v, batch):
        _, decode = batch.attention_backend.get_wrapper()

        if k is not None:
            batch.kv_pool.set_kv_buffer(self.layer_id, batch.kv_pool_ids, k, v)

        o = decode.run(
            q.contiguous().view(
                -1, batch.attention_backend.n_qo_heads, batch.attention_backend.head_dim
            ),
            batch.kv_pool.get_kv_buffer(self.layer_id),
        )

        return o.view(
            -1, batch.attention_backend.n_qo_heads * batch.attention_backend.head_dim
        )
