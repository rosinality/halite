from slickconf import MainConfig, Instance

from halite.projects.common.config import Model, Data, Training, Output


class DiTTraining(Training):
    ema: float = 0.9999
    pretuning_epoch: int = 0
    pretuning_scheduler: Instance | None = None
    freeze_encoder: bool = False


class DiTOutput(Output):
    sampling_step: int


class DiTConfig(MainConfig):
    model: Model
    data: Data
    training: DiTTraining
    output: DiTOutput
