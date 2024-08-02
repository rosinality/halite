import torch
from torch import nn
from torch.nn import functional as F

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.rms_norm import dropout_add_rms_norm

except ImportError:
    pass

# from meshfn.nn.parallel.strategy import Module


class PreNormalization(nn.Module):
    def __init__(self, normalization, module, dropout=0):
        super().__init__()

        self.normalization = normalization
        self.module = module
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, *args, **kwargs):
        out = self.normalization(input)
        out = self.module(out, *args, **kwargs)

        rest = None
        if isinstance(out, tuple):
            out, *rest = out

        out = input + self.dropout(out)

        if rest is not None:
            return (out, *rest)

        return out

    @staticmethod
    def parallelize():
        return {"module": {"_strategy": Module}}


class FusedPreNormalization(nn.Module):
    def __init__(self, normalization, module, dropout=0, rms_norm=False):
        super().__init__()

        self.normalization = normalization
        self.module = module
        self.dropout = nn.Dropout(dropout)
        self.norm_fn = dropout_add_rms_norm if rms_norm else dropout_add_layer_norm

    def forward(self, input, residual=None, *args, **kwargs):
        out, residual = self.norm_fn(
            input,
            residual,
            self.normalization.weight,
            self.normalization.bias,
            self.dropout.p if self.training else 0,
            self.normalization.eps,
            prenorm=True,
        )
        out = self.module(out, *args, **kwargs)

        rest = None
        if isinstance(out, tuple):
            out, *rest = out

        if rest is not None:
            return (out, residual, *rest)

        return out, residual


class FusedPostBlock(nn.Module):
    def __init__(
        self, dim, eps=1e-5, dropout=0, rms_norm=False, device=None, dtype=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

        self.reset_parameters()

        self.eps = eps
        self.dropout = nn.Dropout(dropout)
        self.norm_fn = dropout_add_rms_norm if rms_norm else dropout_add_layer_norm

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, residual):
        out = self.norm_fn(
            input,
            residual,
            self.weight,
            self.bias,
            self.dropout.p if self.training else 0,
            self.eps,
            prenorm=False,
        )

        return out


class LinearLayer(nn.Sequential):
    pass


class NormalizedEmbedding(nn.Embedding):
    def forward(self, input):
        weight = F.normalize(self.weight, dim=1, p=2, eps=1e-6)

        return F.embedding(input, weight, self.padding_idx, self.max_norm,
                        self.norm_type, self.scale_grad_by_freq, self.sparse)

class NormalizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, dim=0):
        super().__init__(in_features, out_features, bias, device, dtype)

        self.dim = dim

    def forward(self, input):
        weight = F.normalize(self.weight, dim=self.dim, p=2, eps=1e-6)

        return F.linear(input, weight, self.bias)


class Scale(nn.Module):
    def __init__(self, dim, init, scale, rescale=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.full((dim,), fill_value=scale, dtype=torch.float32)
        )
        self.init = init
        self.scale = scale
        self.rescale = rescale

    def forward(self, input):
        if self.rescale is None:
            return input * (self.init / self.scale * self.weight)

        return input * (self.init / self.scale * self.rescale * self.weight)


class L2Norm(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()

        self.dim = dim
        self.eps = eps

    def forward(self, input):
        return F.normalize(input, dim=self.dim, p=2, eps=self.eps)
