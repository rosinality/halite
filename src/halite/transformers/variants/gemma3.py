import torch
from torch import nn

from halite.transformers.embedding import TextEmbedding as _TextEmbedding


class TextEmbedding(_TextEmbedding):
    def __init__(
        self,
        embed_input_ids,
        dropout=0,
        embed_init=None,
        multiplier=None,
    ):
        super().__init__(embed_input_ids, dropout, embed_init)

        del self.multiplier
        self.register_buffer("multiplier", torch.tensor(multiplier), persistent=False)

    def forward(self, input_ids):
        word = self.embed_input_ids(input_ids)

        if self.multiplier is not None:
            word = word * self.multiplier.to(word.dtype)

        out = self.dropout(word)

        return out


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, input):
        input_fp32 = input.to(torch.float32)
        out = input_fp32 * torch.rsqrt(
            input_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        out = out * (1.0 + self.weight.float())

        return out.type_as(input)


class MultiModalProjector(nn.Module):
    def __init__(
        self,
        encoder,
        vision_dim,
        text_dim,
        norm_eps,
        image_size,
        patch_size,
        tokens_per_image,
    ):
        super().__init__()

        self.encoder = encoder

        self.input_proj = nn.Parameter(torch.zeros(vision_dim, text_dim))
        self.soft_embed_norm = RMSNorm(vision_dim, eps=norm_eps)

        self.patches_per_image = int(image_size // patch_size)
        self.token_per_side = int(tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.token_per_side
        self.avg_pool = nn.AvgPool2d(self.kernel_size, stride=self.kernel_size)

    def forward(self, images):
        out = self.encoder(images=images).last_hidden_state
        out_dtype = out.dtype

        batch, length, dim = out.shape
        out = out.transpose(1, 2)
        out = out.reshape(batch, dim, self.patches_per_image, self.patches_per_image)
        out = out.contiguous()

        out = self.avg_pool(out).flatten(2).transpose(1, 2)

        out = self.soft_embed_norm(out)
        out = out @ self.input_proj

        return out.to(out_dtype)
