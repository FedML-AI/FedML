import logging
from typing import List, Tuple, Dict
import torch
import copy

from ...core.common.ml_engine_backend import MLEngineBackend

from .agg_operator import FedMLAggOperator
from .base_server_optimizer import ServerOptimizer

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
        """
        1. Return init params_to_client_optimizer for special aggregator need.
        """
        params_to_client_optimizer = dict()
        return params_to_client_optimizer

    def agg(self, args, raw_client_model_or_grad_list):
        """
        Use this function to obtain the final global model.
        """
        # Replace the weight of average.
        for idx in range(len(raw_client_model_or_grad_list)):
            raw_client_model_or_grad_list[idx] = \
                (self.params_to_server_optimizer_dict[idx]["tau_eff"], raw_client_model_or_grad_list[idx][1])

        avg_grads = FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)
        w_global = self.model.state_dict()
        for k in w_global.keys():
            if self.args.gmf != 0:
                if k not in self.global_momentum_buffer:
                    buf = self.global_momentum_buffer[k] = torch.clone(
                        avg_grads[k]).detach()
                    buf.div_(self.args.learning_rate)
                else:
                    buf = self.global_momentum_buffer[k]
                    buf.mul_(self.args.gmf).add_(1 / self.args.learning_rate, avg_grads[k])
                w_global[k].sub_(self.args.learning_rate, buf)
            else:
                w_global[k].sub_(avg_grads[k])
        return w_global


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




    






