from meshfn.transformers.builder.block import TransformerEncoderBlock


class vLLMTransformerEncoderBlock(TransformerEncoderBlock):
    def forward(self, input, position_ids, kv_cache, input_metadata, cache_event):
        out = self.self_attention(
            input, position_ids, kv_cache, input_metadata, cache_event
        )
        out = self.ff(out)

        return out
