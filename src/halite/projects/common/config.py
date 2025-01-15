import copy
import os
import json
from typing import Any

from pydantic import StrictBool, StrictInt, StrictStr
from slickconf import Config, Instance, exempt, deserialize, field

from halite.transformers import TransformerConfig


class Model(Config):
    model_conf: TransformerConfig | None = None
    model: Instance | None = None
    model_infer: Instance | None = None
    tokenizer: Instance | None = None
    parallelize: Instance | None = None
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

        if "parallelize" in conf:
            self.parallelize = Instance(conf["parallelize"])

        tokenizer_path = os.path.join(self.checkpoint_path, "tokenizer.model")
        if os.path.exists(tokenizer_path):
            self.tokenizer.model_path = tokenizer_path

        if "model_conf" in conf:
            self.model_conf = TransformerConfig(**conf["model_conf"])


@exempt
def load_model(checkpoint_path):
    model_conf = deserialize(os.path.join(checkpoint_path, "config.json"))

    conf = field()

    if "model" in model_conf:
        conf.model = model_conf["model"]

    if "model_infer" in model_conf:
        conf.model_infer = model_conf["model_infer"]

    if "tokenizer" in model_conf:
        conf.tokenizer = model_conf["tokenizer"]

    if "parallelize" in model_conf:
        conf.parallelize = model_conf["parallelize"]

    tokenizer_path = os.path.join(checkpoint_path, "tokenizer.model")
    if os.path.exists(tokenizer_path):
        conf.tokenizer.model_path = tokenizer_path

    if "model_conf" in model_conf:
        conf.model_conf = model_conf["model_conf"]

    return conf


def get_tokenizer(model):
    if "tokenizer" in model and model["tokenizer"] is not None:
        tokenizer = copy.deepcopy(model["tokenizer"])

        if "checkpoint_path" in model:
            tokenizer["model_path"] = os.path.join(
                model["checkpoint_path"], "tokenizer.model"
            )

        return Instance(tokenizer)

    if "checkpoint_path" not in model or model["checkpoint_path"] is None:
        raise ValueError("tokenizer or checkpoint is not specified")

    with open(
        os.path.join(model["checkpoint_path"], "tokenizer_config.json"), "r"
    ) as f:
        conf = json.load(f)

    tokenizer = Instance(conf)
    tokenizer_path = os.path.join(model["checkpoint_path"], "tokenizer.model")
    if os.path.exists(tokenizer_path):
        tokenizer.model_path = tokenizer_path

    return tokenizer


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
