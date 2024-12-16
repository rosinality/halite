from pydantic import BaseModel, ConfigDict, StrictInt


class TransformerConfig(BaseModel):
    vocab_size: StrictInt
    dim: StrictInt
    n_layers: StrictInt
    n_heads: StrictInt
    head_dim: StrictInt
    n_key_value_heads: StrictInt
    intermediate_size: StrictInt
    context_len: StrictInt

    model_config = ConfigDict(extra="allow")
