from halite.transformers.block import TransformerEncoderBlock


class InferTransformerEncoderBlock(TransformerEncoderBlock):
    def forward(self, input, batch, residual=None, pos_emb=None):
        out = self.self_attention(input, batch, pos_emb)
        out = self.ff(out)

        rest = None
        if isinstance(out, tuple):
            out, rest = out

        return out, residual, rest
