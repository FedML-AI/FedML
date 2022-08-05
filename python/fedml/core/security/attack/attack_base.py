from abc import ABC, abstractmethod
from typing import Dict


class BaseAttackMethod(ABC):
    def attack_model(
        self, model_list: Dict, global_weights: Dict, refs=None
    ) -> (Dict, Dict):
        pass

    def poison_data(self, dataset):
        pass

    def reconstruct(self, local_w, global_w, refs=None):
        pass
