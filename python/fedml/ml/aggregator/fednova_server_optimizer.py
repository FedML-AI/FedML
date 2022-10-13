import logging
from typing import List, Tuple, Dict
import torch
import copy

from ...core.common.ml_engine_backend import MLEngineBackend

from .agg_operator import FedMLAggOperator
from .base_server_optimizer import ServerOptimizer

from fedml.ml.ml_message import MLMessage


class FedNovaServerOptimizer(ServerOptimizer):
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


    def get_init_params(self) -> Dict:
        other_result = dict()
        return other_result


    def global_agg_seq(self, args, client_result):
        """ Used in hiearchical and sequentially aggregation. """
        key_op_weight_list = [(MLMessage.MODEL_PARAMS, "sum", client_result[MLMessage.MODEL_PARAMS]["agg_weight"])]
        self.global_seq_agg_params(client_result, key_op_weight_list)
        if "tau_eff" not in self.agg_params_dict:
            self.agg_params_dict["tau_eff"] = 0.0
        self.agg_params_dict["tau_eff"] += client_result["tau_eff"]
        return key_op_weight_list


    def agg_seq(self, args, index, client_result, sample_num, training_num_in_round):
        """ Used in hiearchical and sequentially aggregation. """
        key_op_weight_list = [(MLMessage.MODEL_PARAMS, "sum", sample_num/training_num_in_round)]
        self.seq_agg_params(client_result, key_op_weight_list)
        if "tau_eff" not in self.agg_params_dict:
            self.agg_params_dict["tau_eff"] = 0.0
        self.agg_params_dict["tau_eff"] += client_result["tau_eff"].item()
        return key_op_weight_list


    def end_agg_seq(self, args):
        """ Used in hiearchical and sequentially aggregation. """
        key_op_list = [(MLMessage.MODEL_PARAMS, "sum")]
        agg_params_dict = self.end_seq_agg_params(args, key_op_list)
        return agg_params_dict


    def agg(self, args, raw_client_model_or_grad_list):
        """
        Use this function to obtain the final global model.
        """
        if hasattr(self.args, "aggregate_seq") and self.args.aggregate_seq:
            agg_params_dict = self.end_agg_seq(args)
            cum_grad = agg_params_dict[MLMessage.MODEL_PARAMS]["agg_params"]
            tau_eff_sum = agg_params_dict["tau_eff"]
        else:
            cum_grad = FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list, op="sum")
            tau_eff_sum = 0
            for idx in range(len(raw_client_model_or_grad_list)):
                tau_eff_sum += self.client_result_dict[idx]["tau_eff"].item()
        w_global = self.model.cpu().state_dict()
        for k in w_global.keys():
            if self.args.gmf != 0:
                if k not in self.global_momentum_buffer:
                    buf = self.global_momentum_buffer[k] = torch.clone(
                        cum_grad[k]*tau_eff_sum).detach()
                    buf.div_(self.args.learning_rate)
                else:
                    buf = self.global_momentum_buffer[k]
                    buf.mul_(self.args.gmf).add_(1 / self.args.learning_rate, cum_grad[k]*tau_eff_sum)
                w_global[k].sub_(self.args.learning_rate, buf)
            else:
                w_global[k].sub_(cum_grad[k]*tau_eff_sum)
        self.model.load_state_dict(w_global)
        return w_global


    def before_agg(self, client_result_dict, sample_num_dict):
        self.client_result_dict = client_result_dict


    def end_agg(self) -> Dict:
        self.initialize_params_dict()
        self.agg_params_dict["tau_eff"] = 0.0
        other_result = dict()
        return other_result




    






