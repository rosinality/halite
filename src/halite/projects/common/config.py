import os
import json
from typing import Any

from pydantic import StrictBool, StrictInt, StrictStr
from slickconf import Config, Instance

from halite.transformers import TransformerConfig


class Model(Config):
    model_conf: TransformerConfig | None = None
    model: Instance | None = None
    model_infer: Instance | None = None
    tokenizer: Instance | None = None
    wrapper: Instance | None = None
    policy: dict[str, Any] | None = None
    checkpoint_path: StrictStr | None = None

    def model_post_init(self, __context):
        if self.checkpoint_path is None:
            return

        with open(os.path.join(self.checkpoint_path, "config.json"), "r") as f:
            conf = json.load(f)

        if "model" in conf:
            self.model = Instance(conf["model"])

        if "model_infer" in conf:
            self.model_infer = Instance(conf["model_infer"])

        if "tokenizer" in conf:
            self.tokenizer = Instance(conf["tokenizer"])

        tokenizer_path = os.path.join(self.checkpoint_path, "tokenizer.model")
        if os.path.exists(tokenizer_path):
            self.tokenizer.model_path = tokenizer_path

        if "model_conf" in conf:
            self.model_conf = TransformerConfig(**conf["model_conf"])


class Training(Config):
    train_batch_size: StrictInt | None = None
    eval_batch_size: StrictInt | None = None

    max_iter: StrictInt | None = None
    gradient_checkpointing: StrictBool = False
    optimizer: Instance
    scheduler: Instance
    criterion: Instance | None = None
    model_initializer: Instance | None = None
    postprocess: Instance | None = None
    weight_decay: float = 0.0
    clip_grad_norm: float | None = None
    n_epochs: int = 1
    eval_step: int = 1000

    data_parallel_replicate: StrictInt = 1
    data_parallel_shard: StrictInt = 1
    tensor_parallel: StrictInt = 1
    pipeline_parallel: StrictInt = 1


class Dataset(Config):
    root: str
    ratios: dict[str, float]
    shards: dict[str, dict[str, int]]


class Data(Config):
    train: Dataset | None = None
    eval: Dataset | None = None
    train_ratio: float = 1.0
    eval_ratio: float = 1.0
    preprocess: list[Instance] | None = None
    preprocess_eval: list[Instance] | None = None
    collate_fn: Instance | None = None
    collate_fn_eval: Instance | None = None


class Output(Config):
    # print/wandb logging frequencies
    log_step: StrictInt

    # you can set this to save checkpoints at every specified steps.
    save_step: StrictInt | None = None

    # path to the directory which output checkpoint would be saved.
    output_dir: StrictStr | None = None


class Experiment(Config):
    conf: dict[str, Any] | None = None
    tags: dict[str, Any] | None = None
