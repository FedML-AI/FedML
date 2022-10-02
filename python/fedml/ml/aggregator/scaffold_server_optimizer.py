import logging
from typing import List, Tuple, Dict
import torch
import copy

from ...core.common.ml_engine_backend import MLEngineBackend

from .agg_operator import FedMLAggOperator
from .base_server_optimizer import ServerOptimizer

class ScaffoldServerOptimizer(ServerOptimizer):
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
        # self.c_model_global = copy.deepcopy(model).cpu()        
        # for name, params in self.c_model_global.named_parameters():
        #     params.data = params.data*0
        self.c_model_global = {}
        for name, params in model.named_parameters():
            self.c_model_global[name] = copy.deepcopy(params.data*0)


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
        total_weights_delta = FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)
        w_global = self.model.state_dict()
        for key in w_global.keys():
            w_global[key] += total_weights_delta[key] * self.args.server_lr
        # self.model.load_state_dict(w_global)
        return w_global


    def before_agg(self, sample_num_dict):
        key_op_list = [("c_delta_para", "avg")]
        agg_params_dict = self.sync_agg_params(sample_num_dict, key_op_list)
        total_c_delta_para = agg_params_dict["c_delta_para"]

        # c_global_para = self.c_model_global.state_dict()
        for key in self.c_model_global.keys():
            if self.c_model_global[key].type() == 'torch.LongTensor':
                self.c_model_global[key] += total_c_delta_para[key].type(torch.LongTensor)
            elif self.c_model_global[key].type() == 'torch.cuda.LongTensor':
                self.c_model_global[key] += total_c_delta_para[key].type(torch.cuda.LongTensor)
            else:
                self.c_model_global[key] += total_c_delta_para[key]



    def end_agg(self) -> Dict:
        """
        1. Clear self.params_to_server_optimizer_dict 
        2. Return params_to_client_optimizer for special aggregator need.
        """
        self.initialize_params_dict()
        params_to_client_optimizer = dict()
        params_to_client_optimizer["c_model_global"] = self.c_model_global
        return params_to_client_optimizer




    






