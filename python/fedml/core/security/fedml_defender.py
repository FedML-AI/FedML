import logging
from .common.utils import get_total_sample_num
from .defense.geometric_median_defense import GeometricMedianDefense
from .defense.krum_defense import KrumDefense
from .defense.robust_learning_rate_defense import RobustLearningRateDefense
from .defense.slsgd_defense import SLSGDDefense
from ...core.security.defense.norm_diff_clipping_defense import NormDiffClippingDefense
from ...core.security.constants import (
    DEFENSE_NORM_DIFF_CLIPPING,
    DEFENSE_ROBUST_LEARNING_RATE,
    DEFENSE_KRUM,
    DEFENSE_SLSGD,
    DEFENSE_GEO_MEDIAN,
    DEFENSE_CCLIP,
)
from typing import List, Tuple, Dict, Any, Callable


class FedMLDefender:

    _defender_instance = None

    @staticmethod
    def get_instance():
        if FedMLDefender._defender_instance is None:
            FedMLDefender._defender_instance = FedMLDefender()

        return FedMLDefender._defender_instance

    def __init__(self):
        self.is_enabled = False
        self.defense_type = None
        self.defender = None

    def init(self, args):
        if hasattr(args, "enable_defense") and args.enable_defense:
            self.is_enabled = True
            self.defense_type = args.defense_type.strip()
            logging.info("self.defense_type = {}".format(self.defense_type))
            self.defender = None
            if self.defense_type == DEFENSE_NORM_DIFF_CLIPPING:
                self.defender = NormDiffClippingDefense(args)
            elif self.defense_type == DEFENSE_ROBUST_LEARNING_RATE:
                self.defender = RobustLearningRateDefense(args)
            elif self.defense_type == DEFENSE_KRUM:
                self.defender = KrumDefense(args)
            elif self.defense_type == DEFENSE_SLSGD:
                self.defender = SLSGDDefense(args)
            elif self.defense_type == DEFENSE_GEO_MEDIAN:
                self.defender = GeometricMedianDefense(args)
            else:
                raise Exception("args.attack_type is not defined!")
        else:
            self.is_enabled = False

    def is_defense_enabled(self):
        return self.is_enabled

    def get_defense_types(self):
        return self.defense_type

    def is_defense_at_gradients(self):
        return self.is_enabled and self.defense_type in [
            DEFENSE_NORM_DIFF_CLIPPING,
            DEFENSE_KRUM,
            # DEFENSE_SLSGD,
            DEFENSE_GEO_MEDIAN,
        ]

    def is_defense_at_global_model(self):
        return self.is_enabled and self.defense_type in [DEFENSE_SLSGD, DEFENSE_CCLIP]

    def is_defense_at_aggregation(self):
        return self.is_enabled and self.defense_type in [
            DEFENSE_ROBUST_LEARNING_RATE,
            DEFENSE_SLSGD,
            DEFENSE_GEO_MEDIAN,
        ]

    def defend(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        return self.defender.run(
            self, raw_client_grad_list, base_aggregation_func, extra_auxiliary_info
        )

