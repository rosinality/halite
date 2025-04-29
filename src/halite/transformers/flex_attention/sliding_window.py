from halite.transformers.flex_attention import FlexAttentionUpdateMode


class SlidingWindowCausalMask:
    inputs: tuple[str] = ()
    update_mode: FlexAttentionUpdateMode = FlexAttentionUpdateMode.NEVER
    head_shared: bool = True
    batch_shared: bool = True

    def __init__(self, window_size: int):
        self.window_size = window_size

    def __call__(self):
        def sliding_window_causal_mask(b, h, q_idx, kv_idx):
            return (q_idx - kv_idx < self.window_size) & (q_idx >= kv_idx)

        return sliding_window_causal_mask
