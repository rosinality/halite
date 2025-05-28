from pydantic import StrictBool, StrictInt
from slickconf import Config, Instance, MainConfig

from halite.projects.common.config import Model, Data, Output


class Report(Config):
    input_key: str
    output_key: str
    reward_key: str
    additional_keys: list[str] | None = None
    show_every_nth_sample: int = 10
    show_n_samples: int = 3
    log_step: int = 100


class Inference(Config):
    memory_fraction: float = 0.6


class PPO(Config):
    actor: Model
    trainer: Instance
    rollout_generator: Instance
    request_builder: Instance
    inference: Inference | None = None
    ref: Model | None = None
    critic: Model | None = None
    actor_wrapper: Instance | None = None
    infer_dtype: str = "bfloat16"
    report: Report | None = None

    def model_post_init(self, __context):
        if self.inference is None:
            self.inference = Inference()


class Training(Config):
    train_batch_size: StrictInt | None = None
    eval_batch_size: StrictInt | None = None

    ppo_minibatch_size: StrictInt | None = None
    ppo_microbatch_size: StrictInt | None = None
    ppo_n_epochs: StrictInt = 1

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

    train_step_fn: Instance | None = None
    eval_step_fn: Instance | None = None

    data_parallel_replicate: StrictInt = 1
    data_parallel_shard: StrictInt = 1
    tensor_parallel: StrictInt = 1
    pipeline_parallel: StrictInt = 1

    calc_loss_in_model: StrictBool = False


class PPOConfig(MainConfig):
    ppo: PPO
    data: Data
    training: Training
    output: Output
