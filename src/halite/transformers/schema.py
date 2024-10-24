from slickconf import L, field

from halite.transformers.attention import SelfAttention
from halite.transformers.block import TransformerEncoderBlock
from halite.transformers.embedding import TextEmbedding
from halite.transformers.feedforward import FeedForward, VocabParallelLinear
from halite.transformers.normalization import PreNormalization


def plain_transformer_encoder_schema(parallel_head=True):
    schema = field(
        embedding=L[TextEmbedding](),
        blocks=[
            L[TransformerEncoderBlock](
                self_attention=L[PreNormalization](module=L[SelfAttention]()),
                ff=L[PreNormalization](module=L[FeedForward]()),
            )
        ],
    )

    if parallel_head:
        schema.head = L[VocabParallelLinear]()

    return schema.to_dict()
