import logging
from typing import List, Tuple, Dict
import torch
import copy

from ...core.common.ml_engine_backend import MLEngineBackend

from .agg_operator import FedMLAggOperator
from .base_server_optimizer import ServerOptimizer

from fedml.ml.ml_message import MLMessage

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
        other_result = dict()
        return other_result


    def global_agg_seq(self, args, client_result):
        """ Used in hiearchical and sequentially aggregation. """
        key_op_weight_list = [(MLMessage.MODEL_PARAMS, "sum", client_result[MLMessage.MODEL_PARAMS]["agg_weight"])]
        self.global_seq_agg_params(client_result, key_op_weight_list)
        return key_op_weight_list


    def agg_seq(self, args, index, client_result, sample_num, training_num_in_round):
        """
        Use this function to obtain the final global model.
        """
        key_op_weight_list = [(MLMessage.MODEL_PARAMS, "sum", 1.0)]
        self.seq_agg_params(client_result, key_op_weight_list)
        return key_op_weight_list


    def end_agg_seq(self, args):
        key_op_list = [(MLMessage.MODEL_PARAMS, "sum")]
        agg_params_dict = self.end_seq_agg_params(args, key_op_list)
        return agg_params_dict


    def agg(self, args, raw_client_model_or_grad_list):
        """
        Use this function to obtain the final global model.
        """
        if hasattr(self.args, "aggregate_seq") and self.args.aggregate_seq:
            agg_params_dict = self.end_agg_seq(args)
            sum_weights = copy.deepcopy(agg_params_dict[MLMessage.MODEL_PARAMS]["agg_params"])
        else:
            sum_weights = FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list, op="sum")
        model_delta = {}
        w_global = self.model.state_dict()
        for key in sum_weights.keys():
            # https://github.com/TsingZ0/PFL-Non-IID/blob/fd23a2124265fac69c137b313e66e45863487bd5/system/flcore/clients/clientdyn.py#L54
            model_delta[key] = (sum_weights[key] - w_global[key] * self.args.client_num_per_round) / self.args.client_num_in_total
            self.server_state[key] -= self.args.feddyn_alpha * model_delta[key]
            sum_weights[key] = sum_weights[key] / self.args.client_num_per_round
            sum_weights[key] -= (1/self.args.feddyn_alpha) * self.server_state[key]
            # sum_weights[key] -= self.server_state[key]

        return sum_weights


    def before_agg(self, client_result_dict, sample_num_dict):
        self.client_result_dict = client_result_dict

    def end_agg(self) -> Dict:
        self.initialize_params_dict()
        other_result = dict()
        return other_result




    






