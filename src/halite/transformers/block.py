from torch import nn
from torch.distributed.tensor.parallel import SequenceParallel


class TransformerEncoderBlock(nn.Module):
    def __init__(self, self_attention, ff):
        super().__init__()

        self.self_attention = self_attention
        self.ff = ff

    def forward(
        self,
        input,
        residual=None,
        attention_mask=None,
        attention_bias=None,
        pos_emb=None,
        cache=None,
        use_cache=True,
        unpad_params=None,
    ):
        out, next_cache = self.self_attention(
            input,
            attention_mask,
            attention_bias,
            pos_emb,
            cache,
            use_cache,
            unpad_params,
        )
        out = self.ff(out)

        rest = None
        if isinstance(out, tuple):
            out, rest = out

        return out, residual, next_cache, rest


class ResidualBlock(nn.Module):
    def __init__(self, pre_norm, module, post_norm=None, dropout=0):
        super().__init__()

        self.pre_norm = pre_norm
        self.module = module
        self.post_norm = post_norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, *args, **kwargs):
        if self.pre_norm is not None:
            out = self.pre_norm(input)

        out = self.module(out, *args, **kwargs)

        rest = None
        if isinstance(out, tuple):
            out, *rest = out

        if self.post_norm is not None:
            out = self.post_norm(out)

        out = input + self.dropout(out)

        if rest is not None:
            return (out, *rest)

        return out

    def parallelize_plan(self, **kwargs):
        plan = {}

        if self.pre_norm is not None:
            plan["pre_norm"] = SequenceParallel()

        if self.post_norm is not None:
            plan["post_norm"] = SequenceParallel(use_local_output=True)

        return plan


class NGPTBlock(nn.Module):
    def __init__(self, module, post_norm=None, skip_norm=None, scale=None, dropout=0):
        super().__init__()

        self.module = module
        self.post_norm = post_norm
        self.skip_norm = skip_norm
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, *args, **kwargs):
        out = self.module(input, *args, **kwargs)

        rest = None
        if isinstance(out, tuple):
            out, *rest = out

        if self.post_norm is not None:
            out = self.post_norm(out)

        out = out - input

        if self.scale is not None:
            out = self.scale(out)

        out = input + self.dropout(out)

        if self.skip_norm is not None:
            out = self.skip_norm(out)

        if rest is not None:
            return (out, *rest)

        return out

    def parallelize_plan(self, **kwargs):
        plan = {}

        if self.pre_norm is not None:
            plan["pre_norm"] = SequenceParallel()

        if self.post_norm is not None:
            plan["post_norm"] = SequenceParallel(use_local_output=True)

        return plan


class FusedTransformerEncoderBlock(nn.Module):
    def __init__(self, self_attention, ff):
        super().__init__()

        self.self_attention = self_attention
        self.ff = ff

    def forward(
        self,
        input,
        residual=None,
        attention_mask=None,
        attention_bias=None,
        pos_emb=None,
        cache=None,
        use_cache=True,
        unpad_params=None,
    ):
        out, residual, next_cache = self.self_attention(
            input,
            residual,
            attention_mask,
            attention_bias,
            pos_emb,
            cache,
            use_cache,
            unpad_params,
        )
        out, residual = self.ff(out, residual)

        return out, residual, next_cache
