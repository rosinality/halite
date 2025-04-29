from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature

from halite.transformers.flex_attention import FlexAttentionUpdateMode


class BidirectionalMask:
    inputs: tuple[str] = ("bidirection_ids",)
    update_mode: FlexAttentionUpdateMode = FlexAttentionUpdateMode.BATCH
    head_shared: bool = True
    batch_shared: bool = False

    def __init__(self, mask_mod: _mask_mod_signature):
        # get mask mod function, currently does not support mod function with args
        self.mask_mod_fn = mask_mod()

    def __call__(self, bidirection_ids: Tensor):
        """Generates mask mods that apply to inputs to flex attention in the sequence stacked
        format.

        Args:
            mask_mod: The mask mod to apply to the documents
        """

        def bidirectional_mask_mod(b, h, q_idx, kv_idx):
            is_bidir = (bidirection_ids[b, q_idx] > 0) & (
                bidirection_ids[b, q_idx] == bidirection_ids[b, kv_idx]
            )
            inner_mask = self.mask_mod_fn(b, h, q_idx, kv_idx)

            return is_bidir | inner_mask

        return bidirectional_mask_mod
