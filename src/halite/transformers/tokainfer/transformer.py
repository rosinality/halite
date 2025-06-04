import torch

from halite.transformers.transformer import TransformerDecoder
from halite.transformers.tokainfer.attention_fn import create_wrappers
from halite.transformers.tokainfer.kv_cache import LayerKVCache
from halite.transformers.tokainfer.postprocessor import LogitsProcessor
from halite.transformers.tokainfer.types import (
    AttentionInfo,
    BatchState,
    WrapperCollection,
)


class TokaInferTransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        embedding,
        post_embed=None,
        attention_mask=None,
        pos_embed=None,
        blocks=None,
        post_blocks=None,
        head=None,
        tie_embeds=False,
        use_position_ids=False,
        attention_processor="native",
        config=None,
    ):
        super().__init__(
            embedding=embedding,
            post_embed=post_embed,
            attention_mask=attention_mask,
            pos_embed=pos_embed,
            blocks=blocks,
            post_blocks=post_blocks,
            head=head,
            tie_embeds=tie_embeds,
            use_position_ids=use_position_ids,
            attention_processor=attention_processor,
            config=config,
        )

        self.logits_processor = LogitsProcessor()

    def forward_blocks(self, out, batch, attention_mask, pos_emb):
        residual = None

        for index, block in self.blocks.items():
            index = int(index)

            if not self.pos_embed_layer_shared:
                pos_emb = self.get_pos_embed(self.pos_embeds[index], attention_mask)

            out, residual, _ = block(
                out,
                batch,
                pos_emb=pos_emb,
            )

        return out, residual

    @torch.no_grad()
    def forward(self, batch_state: BatchState, async_tp: bool = False):
        self.async_tp = async_tp

        batch = BatchState(
            input_ids=batch_state.input_ids,
            attention_info=batch_state.attention_info,
            position_ids=batch_state.position_ids,
            hidden_states=batch_state.hidden_states,
            lm_head_indices=batch_state.lm_head_indices,
            sampling_params=batch_state.sampling_params,
        )

        out = self.embedding(input_ids=batch.input_ids)

        if self.post_embed is not None:
            out = self.post_embed(out)

        attention_mask = None

        if self.pos_embed_layer_shared:
            pos_emb = self.get_pos_embed(
                self.pos_embed,
                attention_mask,
                batch.position_ids.max() + 1,
                batch.position_ids,
                out.device,
                out.dtype,
            )

        residual = None

        out, residual = self.forward_blocks(out, batch, attention_mask, pos_emb)

        if self.post_blocks is not None:
            if residual is not None:
                out = self.post_blocks(out, residual)

            else:
                out = self.post_blocks(out)

        out = self.logits_processor(batch.input_ids, out, self.head.weight, batch)

        return out

    def set_wrappers(self, wrappers: WrapperCollection):
        self.wrapper_collection = wrappers

        def apply(block):
            if hasattr(block, "wrapper_collection"):
                block.wrapper_collection = wrappers

        self.blocks.apply(apply)

    def setup_caches(self, num_pages: int, page_size: int):
        n_heads, n_key_value_heads, _ = self.get_attention_info()

        wrappers = create_wrappers(
            device=self.device,
            num_attention_heads=n_heads,
            num_key_value_heads=n_key_value_heads,
        )
        self.set_wrappers(wrappers)

        def apply(block):
            if hasattr(block, "layer_cache"):
                block.layer_cache = LayerKVCache(
                    head_dim=block.head_dim,
                    num_kv_heads=block.n_key_value_heads,
                    num_pages=num_pages,
                    page_size=page_size,
                    device=self.device,
                    dtype=self.dtype,
                )

        self.blocks.apply(apply)

    def get_attention_info(self):
        n_heads = None
        n_key_value_heads = None
        head_dim = None

        for block in self.blocks.modules():
            if hasattr(block, "n_heads"):
                n_heads = block.n_heads

            if hasattr(block, "n_key_value_heads"):
                n_key_value_heads = block.n_key_value_heads

            if hasattr(block, "head_dim"):
                head_dim = block.head_dim

            if (
                n_heads is not None
                and n_key_value_heads is not None
                and head_dim is not None
            ):
                break

        return n_heads, n_key_value_heads, head_dim

    def plan(self, attn_info: AttentionInfo, non_blocking: bool = False):
        wrappers = self.wrapper_collection

        n_heads, n_key_value_heads, head_dim = self.get_attention_info()

        def apply(block):
            if hasattr(block, "attention_info"):
                block.attention_info = attn_info

        self.blocks.apply(apply)

        page_size = attn_info.page_size
        q_data_type = self.dtype
        kv_data_type = q_data_type

        if (
            prefill_info := attn_info.prefill_info
        ) is not None and prefill_info.num_tokens > 0:
            wrappers.prefill_wrapper.plan(
                qo_indptr=prefill_info.qo_indptr,
                paged_kv_indptr=prefill_info.kv_indptr,
                paged_kv_indices=prefill_info.kv_indices,
                paged_kv_last_page_len=prefill_info.kv_last_page_len,
                num_kv_heads=n_key_value_heads,
                num_qo_heads=n_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                q_data_type=q_data_type,
                kv_data_type=kv_data_type,
                causal=True,
                non_blocking=non_blocking,
            )

        if (
            hydragen_info := attn_info.hydragen_info
        ) is not None and hydragen_info.num_tokens > 0:
            assert hydragen_info.qo_indptr is not None
            wrappers.hydragen_wrapper.plan(
                qo_indptr=hydragen_info.qo_indptr,
                paged_kv_indptr=hydragen_info.kv_indptr,
                paged_kv_indices=hydragen_info.kv_indices,
                paged_kv_last_page_len=hydragen_info.kv_last_page_len,
                num_kv_heads=n_key_value_heads,
                num_qo_heads=n_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                q_data_type=q_data_type,
                kv_data_type=kv_data_type,
                causal=False,
                non_blocking=non_blocking,
            )

        if (
            decode_info := attn_info.decode_info
        ) is not None and decode_info.num_tokens > 0:
            wrappers.decode_wrapper.plan(
                indptr=decode_info.kv_indptr,
                indices=decode_info.kv_indices,
                last_page_len=decode_info.kv_last_page_len,
                num_kv_heads=n_key_value_heads,
                num_qo_heads=n_heads,
                head_dim=head_dim,
                page_size=page_size,
                q_data_type=q_data_type,
                kv_data_type=kv_data_type,
                non_blocking=non_blocking,
            )
