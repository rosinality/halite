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

        self.tie_weight_key = "embed_input_ids.weight"

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


class MultiModalEmbedding(nn.Module):
    def __init__(self, text_embedding, image_embedding, image_mask_token_id):
        super().__init__()

        self.text_embedding = text_embedding
        self.image_embedding = image_embedding
        self.image_mask_token_id = image_mask_token_id

    @property
    def embed_weight(self):
        return self.text_embedding.embed_weight

    def forward(self, input_ids, images=None):
        input_embed = self.text_embedding(input_ids)

        if images is not None:
            image_embed = self.image_embedding(images=images)
            image_mask = (input_ids == self.image_mask_token_id).unsqueeze(-1)
            image_mask = image_mask.expand(input_embed.shape).to(input_embed.device)
            image_embed = image_embed.to(input_embed)
            input_embed = input_embed.masked_scatter(image_mask, image_embed)

        return input_embed


class PatchEmbedding(nn.Module):
    def __init__(
        self, in_dim, out_dim, patch_size, position_embedding=None, use_conv=False
    ):
        super().__init__()

        if use_conv:
            self.patch_embedding = nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding="valid",
            )

        else:
            self.patch_embedding = nn.Linear(in_dim * (patch_size**2), out_dim)

        self.position_embedding = position_embedding

        self.patch_size = patch_size
        self.use_conv = use_conv

    def forward(self, images):
        if self.use_conv:
            out = self.patch_embedding(images).flatten(2).transpose(1, 2)

        else:
            batch, dim, height, width = images.shape
            out = images.reshape(
                batch,
                dim,
                height // self.patch_size,
                self.patch_size,
                width // self.patch_size,
                self.patch_size,
            )
            out = out.permute(0, 2, 4, 1, 3, 5)
            out = out.reshape(batch, -1, dim * (self.patch_size**2))
            out = self.patch_embedding(out)

        if self.position_embedding is not None:
            out = self.position_embedding(out)

        return out
