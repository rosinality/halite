from random import Random
from typing import Any


class Param:
    def sample(self, random: Random):
        raise NotImplementedError


class Uniform(Param):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample(self, random: Random):
        return random.uniform(self.low, self.high)


class Choice(Param):
    def __init__(self, choices: list[Any]):
        self.choices = choices

    def sample(self, random: Random):
        return random.choice(self.choices)
