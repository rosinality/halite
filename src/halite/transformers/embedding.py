from torch import nn

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import RowwiseParallel


class TextEmbedding(nn.Module):
    def __init__(
        self,
        embed_input_ids,
        dropout=0,
        embed_init=None,
        multiplier=None,
    ):
        super().__init__()

        self.embed_input_ids = embed_input_ids
        self.dropout = nn.Dropout(dropout)

        self.embed_init = embed_init
        self.multiplier = multiplier

    def init_weights(self):
        if self.embed_init is not None:
            self.embed_init(self.embed_input_ids.weight)

    def get_input_embeddings(self):
        return self.embed_input_ids

    @property
    def embed_weight(self):
        return self.embed_input_ids.weight

    def forward(self, input_ids):
        word = self.embed_input_ids(input_ids)

        if self.multiplier is not None:
            word = word * self.multiplier

        out = self.dropout(word)

        return out

    def parallelize_plan(self, **kwargs):
        return {
            "embed_input_ids": RowwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            )
        }
