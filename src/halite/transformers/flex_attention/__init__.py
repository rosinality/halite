from enum import Enum


class FlexAttentionUpdateMode(Enum):
    NEVER = "never"
    BATCH = "batch"
    LAYER = "layer"
