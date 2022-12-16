from collections import OrderedDict
from .cross_round_defense import CrossRoundDefense
from .defense_base import BaseDefenseMethod
from typing import List, Tuple, Any
from .three_sigma_krum_defense import ThreeSigmaKrumDefense


class OutlierDetection(BaseDefenseMethod):
    def __init__(self, config):
        self.cross_round_check = CrossRoundDefense(config)
        self.three_sigma_check = ThreeSigmaKrumDefense(config)

    def defend_before_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, OrderedDict]],
            extra_auxiliary_info: Any = None,
    ):
        raw_client_grad_list = self.cross_round_check.defend_before_aggregation(raw_client_grad_list, extra_auxiliary_info)
        if self.cross_round_check.is_attack_existing:
            self.three_sigma_check.set_potential_malicious_clients(self.cross_round_check.get_potential_poisoned_clients())
            raw_client_grad_list = self.three_sigma_check.defend_before_aggregation(raw_client_grad_list, extra_auxiliary_info)
        return raw_client_grad_list

    def get_malicious_client_idxs(self):
        return self.three_sigma_check.get_malicious_client_idxs()