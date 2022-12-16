from abc import ABC
from collections import OrderedDict
from typing import List, Tuple, Dict, Any


class BaseAttackMethod(ABC):
    def attack_model(
        self, raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ) -> (Dict, Dict):
        pass

    def poison_data(self, dataset):
        pass

    def reconstruct_data(self, a_gradient, extra_auxiliary_info: Any = None):
        pass
