from pydantic import StrictBool, StrictInt, StrictStr
from slickconf import Config, Instance, MainConfig

from halite.projects.common.config import Model, Data, Training, Output


class LMConfig(MainConfig):
    model: Model
    data: Data
    training: Training
    output: Output
