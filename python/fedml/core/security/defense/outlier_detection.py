from .cross_round_defense import CrossRoundDefense
from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any
from .three_sigma_krum_defense import ThreeSigmaKrumDefense


class OutlierDetection(BaseDefenseMethod):
    def __init__(self, config):
        self.cross_round_check = CrossRoundDefense(config)
        self.three_sigma_check = ThreeSigmaKrumDefense(config)

    def run(
            self,
            raw_client_grad_list: List[Tuple[float, Dict]],
            base_aggregation_func: Callable = None,
            extra_auxiliary_info: Any = None,
    ):
        grad_list = self.defend_before_aggregation(
            raw_client_grad_list, extra_auxiliary_info
        )
        return self.defend_on_aggregation(
            grad_list, base_aggregation_func, extra_auxiliary_info
        )

    def defend_before_aggregation(
            self,
            raw_client_grad_list: List[Tuple[float, Dict]],
            extra_auxiliary_info: Any = None,
    ):
        client_grad_list = self.cross_round_check.defend_before_aggregation(raw_client_grad_list, extra_auxiliary_info)
        if self.cross_round_check.is_attack_existing:
            client_grad_list = self.three_sigma_check.defend_before_aggregation(client_grad_list, extra_auxiliary_info)
        return client_grad_list