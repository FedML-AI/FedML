from abc import ABC, abstractmethod


class BaseDPMechanism(ABC):
    @abstractmethod
    def compute_noise_with_shape(self, size):
        pass

    @abstractmethod
    def compute_noise(self):
        pass
