import logging
from ..common.ml_engine_backend import MLEngineBackend
from fedml.core.dp.common.constants import DP_NAIVE_LDP, DP_NAIVE_CDP, NBAFL_DP
from fedml.core.dp.solutions.NbAFL import NbAFL
from fedml.core.dp.solutions.naive_cdp import NaiveGlobalDP
from fedml.core.dp.solutions.naive_ldp import NaiveLocalDP


class FedMLDifferentialPrivacy:
    _dp_instance = None

    @staticmethod
    def get_instance():
        if FedMLDifferentialPrivacy._dp_instance is None:
            FedMLDifferentialPrivacy._dp_instance = FedMLDifferentialPrivacy()
        return FedMLDifferentialPrivacy._dp_instance

    def __init__(self):
        self.dp_solution_type = None
        self.dp_solution = None
        self.is_enabled = False

    def init(self, args):
        if hasattr(args, "enable_dp") and args.enable_dp:
            logging.info(
                ".......init dp......." + args.dp_solution_type + "-" + args.dp_solution_type
            )
            self.is_enabled = True
            # if hasattr(args, "accountant_type") and args.accountant_type in ["adding"]:
            #     self.enable_accountant = True
            self.dp_solution_type = args.dp_solution_type.strip()
            logging.info("self.dp_solution = {}".format(self.dp_solution_type))

            print(f"dp_solution_type={self.dp_solution_type}")

            if self.dp_solution_type == DP_NAIVE_LDP:
                self.dp_solution = NaiveLocalDP(args)
            elif self.dp_solution_type == DP_NAIVE_CDP:
                self.dp_solution = NaiveGlobalDP(args)
            elif self.dp_solution_type == NBAFL_DP:
                self.dp_solution = NbAFL(args)
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
        return self.dp_solution.is_local_dp()

    def is_global_dp_enabled(self):
        return self.dp_solution.is_global_dp()

    def is_to_modify_before_adding_local_noise(self):
        return self.is_dp_enabled() and self.dp_solution_type in []

    def is_to_modify_after_adding_local_noise(self):
        return self.is_dp_enabled() and self.dp_solution_type in []

    def is_to_modify_before_adding_global_noise(self):
        return self.is_dp_enabled() and self.dp_solution_type in []

    def is_to_modify_after_adding_global_noise(self):
        return self.is_dp_enabled() and self.dp_solution_type in []

    def before_add_local_noise(self, local_grad: dict):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        return self.dp_solution.before_add_local_noise(local_grad)

    def after_add_local_noise(
        self,
        local_grad: dict,
    ):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        return self.dp_solution.after_add_local_noise(local_grad)

    def before_add_global_noise(self, global_model: dict):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        return self.dp_solution.before_add_global_noise(global_model)

    def after_add_global_noise(self, global_model: dict):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        return self.dp_solution.after_add_global_noise(global_model)

    def add_local_noise(self, local_grad: dict):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        return self.dp_solution.add_local_noise(local_grad)

    def add_global_noise(self, global_model: dict):
        if self.dp_solution is None:
            raise Exception("dp solution is not initialized!")
        return self.dp_solution.add_global_noise(global_model)
