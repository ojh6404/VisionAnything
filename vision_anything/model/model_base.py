from abc import ABC, abstractmethod
from vision_anything.config.config_base import InferenceConfigBase


class InferenceModelBase(ABC):
    def __init__(self, config: InferenceConfigBase):
        self.model_config = config
        self.predictor = config.get_predictor()

    @abstractmethod
    def predict(self, image):
        pass
