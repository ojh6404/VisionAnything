from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class InferenceConfigBase(ABC):
    model_name: str
    device: str = "cuda:0"

    @abstractmethod
    def get_predictor(self):
        pass

    @classmethod
    @abstractmethod
    def from_args(cls):
        pass
