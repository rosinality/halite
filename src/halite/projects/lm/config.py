from pydantic import StrictBool, StrictInt, StrictStr
from slickconf import Config, Instance, MainConfig


class Model(Config):
    model: Instance
    wrapper: Instance | None = None


class Dataset(Config):
    root: str
    ratios: dict[str, float]
    shards: dict[str, dict[str, int]]


class LMData(Config):
    train: Dataset | None = None
    train_ratio: float = 1.0
    eval: Dataset | None = None
    eval_ratio: float = 1.0
    preprocess: list[Instance] | None = None
    collate_fn: Instance | None = None


class Training(Config):
    train_batch_size: StrictInt | None = None
    eval_batch_size: StrictInt | None = None

    max_iter: StrictInt | None = None
    gradient_checkpointing: StrictBool = False
    optimizer: Instance
    scheduler: Instance
    criterion: Instance | None = None
    postprocess: Instance | None = None
    weight_decay: float = 0.0
    clip_grad_norm: float | None = None
    n_epochs: int = 1
    eval_step: int = 1000

    data_parallel_replicate: StrictInt = 1
    data_parallel_shard: StrictInt = 1
    tensor_parallel: StrictInt = 1
    pipeline_parallel: StrictInt = 1


class Output(Config):
    # print/wandb logging frequencies
    log_step: StrictInt

    # you can set this to save checkpoints at every specified steps.
    save_step: StrictInt | None = None

    # path to the directory which output checkpoint would be saved.
    output_dir: StrictStr | None = None


class LMConfig(MainConfig):
    model: Model
    data: LMData
    training: Training
    output: Output
