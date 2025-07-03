import torch

from halite.transformers.transformer import TransformerDecoder
from halite.transformers.infer.attention import FlashInferAttention
from halite.transformers.infer.postprocessor import LogitsProcessor


class InferTransformerDecoder(TransformerDecoder):
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

        self.attention_backend_modules = None
        self.kv_pool_modules = None

    def set_attention_backend(self, attention_backend):
        if self.attention_backend_modules is None:
            self.attention_backend_modules = []

            for module in self.modules():
                if isinstance(module, FlashInferAttention):
                    self.attention_backend_modules.append(module)

        for module in self.attention_backend_modules:
            module.attention_backend = attention_backend

    def set_kv_pool(self, kv_pool):
        if self.kv_pool_modules is None:
            self.kv_pool_modules = []

            for module in self.modules():
                if isinstance(module, FlashInferAttention):
                    self.kv_pool_modules.append(module)

        for module in self.kv_pool_modules:
            module.kv_pool = kv_pool

    @torch.no_grad()
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
    def forward(self, batch):
        out = self.embedding(input_ids=batch.input_ids)

        if self.post_embed is not None:
            out = self.post_embed(out)

        attention_mask = None

        if self.pos_embed_layer_shared:
            pos_emb = self.get_pos_embed(
                self.pos_embed,
                attention_mask,
                batch.seq_lens_max,
                batch.positions,
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
