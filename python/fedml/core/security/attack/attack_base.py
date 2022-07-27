from abc import ABC, abstractmethod
from typing import Dict

from torch.utils.data import DataLoader


class BaseAttackMethod(ABC):
    @abstractmethod
    def attack_model(
        self, local_weights: Dict, global_weights: Dict, refs=None
    ) -> (Dict, Dict):
        pass

    # def poison_data(
    #     self, train_data_loader: DataLoader, test_data_loader: DataLoader
    # ) -> (DataLoader, DataLoader):
    #     pass

    def poison_data(self, dataset):
        pass

    def reconstruct(self, local_w, global_w, refs=None):
        pass
