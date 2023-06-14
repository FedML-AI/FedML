import logging
from collections import OrderedDict
from typing import List, Tuple, Any, Callable
from .defense.RFA_defense import RFADefense
from .defense.coordinate_wise_median_defense import CoordinateWiseMedianDefense
from .defense.coordinate_wise_trimmed_mean_defense import CoordinateWiseTrimmedMeanDefense
from .defense.crfl_defense import CRFLDefense
from .defense.outlier_detection import OutlierDetection
from .defense.three_sigma_defense import ThreeSigmaDefense
from .defense.three_sigma_geomedian_defense import ThreeSigmaGeoMedianDefense
from .defense.three_sigma_krum_defense import ThreeSigmaKrumDefense
from ..common.ml_engine_backend import MLEngineBackend
from .defense.cclip_defense import CClipDefense
from .defense.foolsgold_defense import FoolsGoldDefense
from .defense.geometric_median_defense import GeometricMedianDefense
from .defense.krum_defense import KrumDefense
from .defense.robust_learning_rate_defense import RobustLearningRateDefense
from .defense.slsgd_defense import SLSGDDefense
from .defense.weak_dp_defense import WeakDPDefense
from ...core.security.defense.norm_diff_clipping_defense import NormDiffClippingDefense
from ...core.security.constants import (
    DEFENSE_NORM_DIFF_CLIPPING,
    DEFENSE_ROBUST_LEARNING_RATE,
    DEFENSE_KRUM,
    DEFENSE_SLSGD,
    DEFENSE_GEO_MEDIAN,
    DEFENSE_CCLIP,
    DEFENSE_WEAK_DP,
    DEFENSE_RFA,
    DEFENSE_FOOLSGOLD,
    DEFENSE_THREESIGMA,
    DEFENSE_CRFL,
    DEFENSE_MULTIKRUM,
    DEFENSE_TRIMMED_MEAN,
    DEFENSE_THREESIGMA_GEOMEDIAN,
    DEFENSE_THREESIGMA_KRUM, ANOMALY_DETECTION, DEFENSE_WISE_MEDIAN, DEFENSE_DIFF_CLIPPING,
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
            self.args = args
            logging.info("------init defense..." + args.defense_type)
            self.is_enabled = True
            self.defense_type = args.defense_type.strip()
            logging.info("self.defense_type = {}".format(self.defense_type))
            self.defender = None
            if self.defense_type == DEFENSE_NORM_DIFF_CLIPPING:
                self.defender = NormDiffClippingDefense(args)
            elif self.defense_type == DEFENSE_ROBUST_LEARNING_RATE:
                self.defender = RobustLearningRateDefense(args)
            elif self.defense_type in [DEFENSE_KRUM, DEFENSE_MULTIKRUM]:
                self.defender = KrumDefense(args)
            elif self.defense_type == DEFENSE_SLSGD:
                self.defender = SLSGDDefense(args)
            elif self.defense_type == DEFENSE_GEO_MEDIAN:
                self.defender = GeometricMedianDefense(args)
            elif self.defense_type == DEFENSE_WEAK_DP:
                self.defender = WeakDPDefense(args)
            elif self.defense_type == DEFENSE_CCLIP:
                self.defender = CClipDefense(args)
            elif self.defense_type == DEFENSE_WISE_MEDIAN:
                self.defender = CoordinateWiseMedianDefense(args)
            elif self.defense_type == DEFENSE_RFA:
                self.defender = RFADefense(args)
            elif self.defense_type == DEFENSE_FOOLSGOLD:
                self.defender = FoolsGoldDefense(args)
            elif self.defense_type == DEFENSE_THREESIGMA:
                self.defender = ThreeSigmaDefense(args)
            elif self.defense_type == DEFENSE_THREESIGMA_GEOMEDIAN:
                self.defender = ThreeSigmaGeoMedianDefense(args)
            elif self.defense_type == DEFENSE_THREESIGMA_KRUM:
                self.defender = ThreeSigmaKrumDefense(args)
            elif self.defense_type == DEFENSE_CRFL:
                self.defender = CRFLDefense(args)
            elif self.defense_type == DEFENSE_TRIMMED_MEAN:
                self.defender = CoordinateWiseTrimmedMeanDefense(args)
            elif self.defense_type == ANOMALY_DETECTION:
                self.defender = OutlierDetection(args)
            else:
                raise Exception("args.defense_type is not defined!")
        else:
            self.is_enabled = False

        if (
            self.is_enabled
            and hasattr(args, MLEngineBackend.ml_engine_args_flag)
            and args.ml_engine
            in [
                MLEngineBackend.ml_engine_backend_tf,
                MLEngineBackend.ml_engine_backend_jax,
                MLEngineBackend.ml_engine_backend_mxnet,
            ]
        ):
            logging.info(
                "FedMLDefender is not supported for the machine learning engine: %s. "
                "We will support more engines in the future iteration." % args.ml_engine
            )
            self.is_enabled = False

    def is_defense_enabled(self):
        return self.is_enabled

    def defend(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        return self.defender.run(
            raw_client_grad_list, base_aggregation_func, extra_auxiliary_info
        )

    def is_defense_on_aggregation(self):
        return self.is_defense_enabled() and self.defense_type in [DEFENSE_SLSGD, DEFENSE_RFA, DEFENSE_WISE_MEDIAN, DEFENSE_GEO_MEDIAN]

    def is_defense_before_aggregation(self):
        return self.is_defense_enabled() and self.defense_type in [
            DEFENSE_SLSGD,
            DEFENSE_FOOLSGOLD,
            DEFENSE_THREESIGMA,
            DEFENSE_THREESIGMA_GEOMEDIAN,
            DEFENSE_THREESIGMA_KRUM,
            DEFENSE_KRUM,
            DEFENSE_CCLIP,
            DEFENSE_MULTIKRUM,
            DEFENSE_TRIMMED_MEAN,
            ANOMALY_DETECTION,
            DEFENSE_NORM_DIFF_CLIPPING
        ]

    def is_defense_after_aggregation(self):
        return self.is_defense_enabled() and self.defense_type in [DEFENSE_CRFL, DEFENSE_CCLIP]

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        if self.is_defense_before_aggregation():
            return self.defender.defend_before_aggregation(
                raw_client_grad_list, extra_auxiliary_info
            )
        return raw_client_grad_list

    def defend_on_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        if self.is_defense_on_aggregation():
            return self.defender.defend_on_aggregation(
                raw_client_grad_list, base_aggregation_func, extra_auxiliary_info
            )
        return base_aggregation_func(args=self.args, raw_grad_list=raw_client_grad_list)

    def defend_after_aggregation(self, global_model):
        if self.defender is None:
            raise Exception("defender is not initialized!")
        if self.is_defense_after_aggregation():
            return self.defender.defend_after_aggregation(global_model)
        return global_model

    def get_malicious_client_idxs(self):
        return self.defender.get_malicious_client_idxs()

    def get_benign_client_idxs(self, client_idxs):
        return [i for i in client_idxs if i not in self.defender.get_malicious_client_idxs()]
