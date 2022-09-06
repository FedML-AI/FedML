import logging
from typing import List, Tuple, Dict, Any

from ..common.constants import NBAFL_DP
from ...common.ml_engine_backend import MLEngineBackend
from fedml.core.dp.solutions.NbAFL import NbAFL_DP


class FedMLDP:
    _defender_instance = None

    @staticmethod
    def get_instance():
        if FedMLDP._defender_instance is FedMLDP:
            FedMLDP._defender_instance = FedMLDP()
        return FedMLDP._defender_instance

    def __init__(self):
        self.is_enabled = False
        self.dp_solution = None
        self.dp_type = None

    def init(self, args):
        if hasattr(args, "enable_dp") and args.enable_dp:
            self.is_enabled = True
            logging.info("------init defense..." + args.defense_type)
            self.dp_type = args.dp_type.strip()
            logging.info("self.dp_type = {}".format(self.dp_type))
            self.dp_solution = None
            if self.dp_type == NBAFL_DP:
                self.dp_solution = NbAFL_DP(args)
            else:
                raise Exception("args.dp_type is not defined!")
        else:
            self.is_enabled = False

        if hasattr(args, MLEngineBackend.ml_engine_args_flag) and args.ml_engine in [
            MLEngineBackend.ml_engine_backend_tf,
            MLEngineBackend.ml_engine_backend_jax,
            MLEngineBackend.ml_engine_backend_mxnet,
        ]:
            logging.info(
                "FedMLDP is not supported for the machine learning engine: %s. "
                "We will support more engines in the future iteration." % args.ml_engine
            )
            self.is_enabled = False

    def is_dp_enabled(self):
        return self.is_enabled

    def is_local_dp(self):
        return self.is_dp_enabled() and self.dp_type in [NBAFL_DP]

    def is_global_dp(self):
        return self.is_dp_enabled() and self.dp_type in [NBAFL_DP]

    def add_local_noise(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        extra_auxiliary_info: Any = None,
    ):
        if self.dp_solution is None:
            raise Exception("dp is not initialized!")
        if self.is_local_dp():
            return self.dp_solution.add_local_dp(
                raw_client_grad_list, extra_auxiliary_info
            )
        return raw_client_grad_list

    def add_global_noise(
        self,
        global_model,
        extra_auxiliary_info: Any = None,
    ):
        if self.dp_solution is None:
            raise Exception("defender is not initialized!")
        if self.is_global_dp():
            return self.dp_solution.add_global_dp(
                global_model, extra_auxiliary_info
            )
        return global_model



    def before_add_local_noise(self):
        pass

    def after_add_local_noise(self):
        pass

    def before_add_global_noise(self):
        pass

    def after_add_global_noise(self):
        pass