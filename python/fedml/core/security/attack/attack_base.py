from abc import ABC
from collections import OrderedDict
from typing import List, Tuple, Dict, Any


class BaseAttackMethod(ABC):
    def attack_model(
            self, raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
    ) -> (Dict, Dict):
        pass

    def reconstruct_data(self, a_gradient, extra_auxiliary_info: Any = None):
        pass

    ############### for data poisoning attacks ###############
    def compute_poisoned_client_ids(self, client_ids: List):
        pass

    def is_to_poison_data(self):
        pass

    def poison_data(self, dataset):
        pass
    ############### for data poisoning attacks ###############
