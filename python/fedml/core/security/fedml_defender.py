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
                self.defender = NormDiffClippingDefense(args.norm_bound)
            elif self.defense_type == DEFENSE_ROBUST_LEARNING_RATE:
                self.defender = RobustLearningRateDefense(args.robust_threshold)
            elif self.defense_type == DEFENSE_KRUM:
                self.defender = KrumDefense(args.byzantine_client_num, args.multi)
            elif self.defense_type == DEFENSE_SLSGD:
                self.defender = SLSGDDefense(
                    args.trim_param_b, args.alpha, args.option_type
                )
            elif self.defense_type == DEFENSE_GEO_MEDIAN:
                self.defender = GeometricMedianDefense(
                    args.byzantine_client_num, args.client_num_per_round, args.batch_num
                )
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

    def defend(self, client_grad_list, global_w):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        new_grad_list = self.defender.defend(client_grad_list, global_w)
        training_num = get_total_sample_num(new_grad_list)
        return training_num, new_grad_list

    def robust_aggregate(self, client_grad_list, global_w=None):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        return self.defender.robust_aggregate(client_grad_list, global_w=global_w)

    def robustify_global_model(self, avg_params, previous_global_w=None):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        return self.defender.robustify_global_model(avg_params, previous_global_w=previous_global_w)
