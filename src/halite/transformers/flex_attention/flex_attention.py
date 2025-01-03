from typing import Callable, NamedTuple

from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
)

from halite.transformers.flex_attention import FlexAttentionUpdateMode


class FlexAttentionInput(NamedTuple):
    score_mod: Callable
    block_mask: BlockMask


class FlexAttentionProcessor(nn.Module):
    def __init__(
        self,
        score_mod: Callable = None,
        block_mask: Callable = None,
        n_heads: int | None = None,
        compile: bool = True,
    ):
        super().__init__()

        self.score_mod = score_mod
        self.block_mask = block_mask
        self.n_heads = n_heads

        if self.block_mask is not None and self.block_mask.head_shared:
            self.n_heads = None

        self.score_mod_update_mode = (
            score_mod.update_mode
            if score_mod is not None
            else FlexAttentionUpdateMode.NEVER
        )
        self.block_mask_update_mode = (
            block_mask.update_mode
            if block_mask is not None
            else FlexAttentionUpdateMode.NEVER
        )

        self.score_mod_inputs = score_mod.inputs if score_mod is not None else ()
        self.block_mask_inputs = block_mask.inputs if block_mask is not None else ()

        self.compile = compile

        self._cached_score_mod = None
        self._cached_block_mask = None

    def __call__(self, batch, q_len, kv_len, device="cuda", **kwargs):
        score_mod = None
        block_mask = None

        if (
            self._cached_score_mod is not None
            and self.score_mod_update_mode == FlexAttentionUpdateMode.NEVER
        ):
            score_mod = self._cached_score_mod

        if (
            self._cached_block_mask is not None
            and self.block_mask_update_mode == FlexAttentionUpdateMode.NEVER
        ):
            block_mask = self._cached_block_mask

        kwargs.update(
            {
                "batch": batch,
                "n_heads": self.n_heads,
                "q_len": q_len,
                "kv_len": kv_len,
                "device": device,
            }
        )

        if score_mod is None and self.score_mod is not None:
            score_mod_kwargs = {k: kwargs[k] for k in self.score_mod_inputs}
            score_mod = self.score_mod(**score_mod_kwargs)

        if block_mask is None and self.block_mask is not None:
            block_mask_kwargs = {k: kwargs[k] for k in self.block_mask_inputs}
            block_mask = self.block_mask(**block_mask_kwargs)

        if not isinstance(block_mask, BlockMask):
            block_mask = create_block_mask(
                block_mask,
                batch,
                self.n_heads,
                q_len,
                kv_len,
                device,
                _compile=self.compile,
            )

        if (
            self._cached_score_mod is None
            and self.score_mod_update_mode == FlexAttentionUpdateMode.NEVER
        ):
            self._cached_score_mod = score_mod

        if (
            self._cached_block_mask is None
            and self.block_mask_update_mode == FlexAttentionUpdateMode.NEVER
        ):
            self._cached_block_mask = block_mask

        return FlexAttentionInput(score_mod, block_mask)
