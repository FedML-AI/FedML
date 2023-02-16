import logging
from collections import OrderedDict
from typing import List, Tuple
from fedml.core.dp.common.constants import DP_LDP, DP_CDP, NBAFL_DP, DP_CLIP
from fedml.core.dp.frames.cdp import GlobalDP
from fedml.core.dp.frames.ldp import LocalDP
from .budget_accountant.rdp_accountant import RDP_Accountant
from .frames.NbAFL import NbAFL_DP
from .frames.dp_clip import DP_Clip
from ..common.ml_engine_backend import MLEngineBackend


class FedMLDifferentialPrivacy:
    _dp_instance = None

    @staticmethod
    def get_instance():
        if FedMLDifferentialPrivacy._dp_instance is None:
            FedMLDifferentialPrivacy._dp_instance = FedMLDifferentialPrivacy()
        return FedMLDifferentialPrivacy._dp_instance

    def __init__(self):
        self.enable_rdp_accountant = False
        self.max_grad_norm = None
        self.dp_solution_type = None
        self.dp_solution = None
        self.dp_accountant = None
        self.is_enabled = False
        self.privacy_engine = None
        self.current_round = 0
        self.accountant = None
        self.delta = None

    def init(self, args):
        if hasattr(args, "enable_dp") and args.enable_dp:
            logging.info(".......init dp......." + args.dp_solution_type + "-" + args.dp_solution_type)
            self.is_enabled = True
            self.dp_solution_type = args.dp_solution_type.strip()
            if hasattr(args, "max_grad_norm"):
                self.max_grad_norm = args.max_grad_norm
            self.delta = args.delta

            logging.info("self.dp_solution = {}".format(self.dp_solution_type))

            print(f"dp_solution_type={self.dp_solution_type}")

            if self.dp_solution_type == DP_LDP:
                self.dp_solution = LocalDP(args)
            elif self.dp_solution_type == DP_CDP:
                self.dp_solution = GlobalDP(args)
            elif self.dp_solution_type == NBAFL_DP:
                self.dp_solution = NbAFL_DP(args)
            elif self.dp_solution_type == DP_CLIP:
                self.dp_solution = DP_Clip(args)
            else:
                raise Exception("dp solution is not defined")

        if hasattr(args, MLEngineBackend.ml_engine_args_flag) and args.ml_engine in [
            MLEngineBackend.ml_engine_backend_tf,
            MLEngineBackend.ml_engine_backend_jax,
            MLEngineBackend.ml_engine_backend_mxnet,
        ]:
            logging.info(
                "FedMLDifferentialPrivacy is not supported for the machine learning engine: %s. "
                "We will support more engines in the future iteration." % args.ml_engine
            )
            self.is_enabled = False

    def is_dp_enabled(self):
        return self.is_enabled

    def is_local_dp_enabled(self):
        return self.is_enabled and self.dp_solution_type in [DP_LDP]

    def is_global_dp_enabled(self):
        return self.is_enabled and self.dp_solution_type in [DP_CDP]

    def is_clipping(self):
        return self.is_enabled and self.dp_solution_type in [DP_CDP]

    def to_compute_params_in_aggregation_enabled(self):
        return self.is_enabled and self.dp_solution_type in [NBAFL_DP]

    def global_clip(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        # todo: laplace:
        return self.dp_solution.global_clip(raw_client_model_or_grad_list)

    def add_local_noise(self, local_grad: OrderedDict):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        return self.dp_solution.add_local_noise(local_grad)

    def add_global_noise(self, global_model: OrderedDict):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        if self.dp_solution.is_rdp_accountant_enabled:
            self.dp_solution.accountant.get_epsilon(self.delta)
        return self.dp_solution.add_global_noise(global_model)

    def set_params_for_dp(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        self.dp_solution.set_params_for_dp(raw_client_model_or_grad_list)

