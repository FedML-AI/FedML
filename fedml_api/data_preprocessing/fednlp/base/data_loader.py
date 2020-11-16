from abc import ABC, abstractmethod
from .globals import *

class BaseDataLoader(ABC):
    @abstractmethod
    def __init__(self, data_path):
        self.data_path = data_path
        self.X = []
        self.Y = []

    @abstractmethod
    def data_loader(self, client_idx=None):
        pass

    @abstractmethod
    def process_data(self, file_path):
        pass
