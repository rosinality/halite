from slickconf import MainConfig

from halite.projects.common.config import Model, Data, Training, Output


class DiTTraining(Training):
    ema: float = 0.9999


class DiTOutput(Output):
    sampling_step: int


class DiTConfig(MainConfig):
    model: Model
    data: Data
    training: DiTTraining
    output: DiTOutput
