from meshfn.transformers.builder.transformer import TransformerDecoder
from meshfn.transformers.builder.vllm.sampler import Sampler


class vLLMTransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        embedding,
        post_embed=None,
        attention_mask=None,
        pos_embed=None,
        blocks=None,
        post_blocks=None,
        head=None,
        batch_first=False,
        tie_embeds=False,
        use_position_ids=False,
        parallel_context=None,
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
            batch_first=batch_first,
            tie_embeds=tie_embeds,
            use_position_ids=use_position_ids,
            parallel_context=parallel_context,
            config=config,
        )

        self.sampler = Sampler(self.head.n_vocab, parallel_context)

    def forward(self, input_ids, positions, kv_caches, input_metadata, cache_events):
        out = self.embedding(input_ids=input_ids)

        if self.post_embed is not None:
            out = self.post_embed(out)

        for index, block in enumerate(self.blocks):
            if cache_events is None:
                cache_event = None

            else:
                cache_event = cache_events[index]

            out = block(out, positions, kv_caches[index], input_metadata, cache_event)

        if self.post_blocks is not None:
            out = self.post_blocks(out)

        if self.head is not None:
            head_weight = self.head.weight
            head_bias = self.head.bias

            out = self.sampler(head_weight, out, input_metadata, head_bias)

        return out
