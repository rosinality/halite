from halite.transformers.flex_attention import FlexAttentionUpdateMode


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class CausalMask:
    inputs: tuple[str] = ()
    update_mode: FlexAttentionUpdateMode = FlexAttentionUpdateMode.NEVER
    head_shared: bool = True

    def __call__(self):
        return causal_mask
