from torch import nn

from meshfn.transformers.builder.attention import SelfAttention
from meshfn.transformers.builder.position import alibi_slopes


def vllm_attention():
    from vllm.model_executor.layers import attention

    return attention


class PagedAttentionWithALiBi(nn.Module):
    def __init__(self, n_head, head_dim, normalize=True):
        super().__init__()

        scale = 1

        self.n_head = n_head
        self.head_dim = head_dim
        self.n_key_value_head = 0

        if normalize:
            scale = 1 / (head_dim**0.5)

        self.attention = vllm_attention().PagedAttentionWithALiBi(
            n_head, head_dim, scale, alibi_slopes(n_head)
        )
        # in default alibi slopes registered as a buffer
        # but buffer changes dtype if .to() is called,
        # and change float32 to bfloat16 or float16 will make substantial difference
        del self.attention.alibi_slopes
        self.attention.alibi_slopes = alibi_slopes(n_head)

    def forward(
        self, q, k, v, position_ids, k_cache, v_cache, input_metadata, cache_event
    ):
        return self.attention(q, k, v, k_cache, v_cache, input_metadata, cache_event)


class vLLMSelfAttention(SelfAttention):
    def forward(self, input, position_ids, kv_cache, input_metadata, cache_event):
        qkv = self.qkv(input)
        q, k, v = self.qkv_split(qkv)
        k_cache, v_cache = kv_cache
        out = self.attention(
            q, k, v, position_ids, k_cache, v_cache, input_metadata, cache_event
        )
        out = self.out(out)

        return out
