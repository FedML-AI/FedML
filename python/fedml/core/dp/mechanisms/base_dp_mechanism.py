from abc import ABC, abstractmethod


class BaseDPMechanism(ABC):
    @abstractmethod
    def add_noise(self, size):
        pass
