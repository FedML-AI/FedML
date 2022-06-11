from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @abstractmethod
    def transform(self, *args):
        pass
