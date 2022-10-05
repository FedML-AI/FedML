import logging
from typing import List, Tuple, Dict
import torch
import copy

from ...core.common.ml_engine_backend import MLEngineBackend

from .agg_operator import FedMLAggOperator
from .base_server_optimizer import ServerOptimizer


class FedDynServerOptimizer(ServerOptimizer):
    """Abstract base class for federated learning trainer.
    1. The goal of this abstract class is to be compatible to
    any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
    2. This class is used only on server side.
    3. This class is an operator which can caches states across rounds. 
    Because the server consistently exists.
    """
    def __init__(self, args):
        super().__init__(args)


    def initialize(self, args, model):
        self.model = model
        self.server_state = {
            key: torch.zeros(params.shape)
            for key, params in model.state_dict().items()
        }


    def get_init_params(self) -> Dict:
        """
        1. Return init params_to_client_optimizer for special aggregator need.
        """
        params_to_client_optimizer = dict()
        return params_to_client_optimizer

    def agg(self, args, raw_client_model_or_grad_list):
        """
        Use this function to obtain the final global model.
        """
        sum_weights = FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list, op="sum")
        model_delta = {}
        w_global = self.model.state_dict()
        for key in sum_weights.keys():
            # https://github.com/TsingZ0/PFL-Non-IID/blob/fd23a2124265fac69c137b313e66e45863487bd5/system/flcore/clients/clientdyn.py#L54
            model_delta[key] = (sum_weights[key] - w_global[key] * self.args.client_num_per_round) / self.args.client_num_in_total
            self.server_state[key] -= self.args.feddyn_alpha * model_delta[key]
            sum_weights[key] = sum_weights[key] / self.args.client_num_per_round
            # sum_weights[key] -= (1/self.args.feddyn_alpha) * self.server_state[key]
            sum_weights[key] -= self.server_state[key]

        return sum_weights


    def before_agg(self, sample_num_dict):
        pass

    def end_agg(self) -> Dict:
        """
        1. Clear self.params_to_server_optimizer_dict 
        2. Return params_to_client_optimizer for special aggregator need.
        """
        self.initialize_params_dict()
        params_to_client_optimizer = dict()
        return params_to_client_optimizer




    






