import torch

from halite.transformers.transformer import TransformerDecoder
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

    @torch.inference_mode()
    def forward(self, batch):
        out = self.embedding(input_ids=batch.input_ids)

        if self.post_embed is not None:
            out = self.post_embed(out)

        attention_mask = None

        if self.pos_embed_layer_shared:
            pos_emb = self.get_pos_embed(
                self.pos_embed,
                attention_mask,
                batch.seq_lens.max(),
                batch.positions,
                out.device,
                out.dtype,
            )

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

        if self.post_blocks is not None:
            if residual is not None:
                out = self.post_blocks(out, residual)

            else:
                out = self.post_blocks(out)

        out = self.logits_processor(batch.input_ids, out, self.head.weight, batch)

        return out
