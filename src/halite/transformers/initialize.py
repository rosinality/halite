from torch import nn
from torch.distributed import get_rank

from halite.transformers.normalization import LinearLayer


def init_weights(module, init_fn):
    is_seq = False
    if isinstance(module, LinearLayer):
        module = module[0]
        is_seq = True

    if init_fn is not None:
        init_fn(module.weight)

    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0)
