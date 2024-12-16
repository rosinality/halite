from typing import Any

from slickconf import Config, Instance

from halite.transformers import TransformerConfig


class Model(Config):
    model_conf: TransformerConfig
    model: Instance | None = None
    model_infer: Instance | None = None
    tokenizer: Instance | None = None
    wrapper: Instance | None = None
    policy: dict[str, Any] | None = None
