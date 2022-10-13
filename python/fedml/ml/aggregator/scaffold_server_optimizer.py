import logging
from typing import List, Tuple, Dict
import torch
import copy

from ...core.common.ml_engine_backend import MLEngineBackend

from .agg_operator import FedMLAggOperator
from .base_server_optimizer import ServerOptimizer

from fedml.ml.ml_message import MLMessage

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
        self.model = model
        self.c_model_global = {}
        for name, params in model.named_parameters():
            self.c_model_global[name] = copy.deepcopy(params.data*0)


    def get_init_params(self) -> Dict:
        other_result = dict()
        other_result["c_model_global"] = self.c_model_global
        return other_result


    def global_agg_seq(self, args, client_result):
        """ Used in hiearchical and sequentially aggregation. """
        key_op_weight_list = [(MLMessage.MODEL_PARAMS, "weighted_avg", client_result[MLMessage.MODEL_PARAMS]["agg_weight"]),
                              ("c_delta_para", "avg", client_result[MLMessage.MODEL_PARAMS]["agg_weight"])]
        self.global_seq_agg_params(client_result, key_op_weight_list)
        return key_op_weight_list


    def agg_seq(self, args, index, client_result, sample_num, training_num_in_round):
        """
        Use this function to obtain the final global model.
        """
        key_op_weight_list = [(MLMessage.MODEL_PARAMS, "weighted_avg", sample_num/training_num_in_round),
                              ("c_delta_para", "avg", sample_num/training_num_in_round)]
        self.seq_agg_params(client_result, key_op_weight_list)
        return key_op_weight_list


    def end_agg_seq(self, args):
        key_op_list = [(MLMessage.MODEL_PARAMS, "weighted_avg"),
                       ("c_delta_para", "avg")]
        agg_params_dict = self.end_seq_agg_params(args, key_op_list)
        return agg_params_dict


    def agg(self, args, raw_client_model_or_grad_list):
        """
        Use this function to obtain the final global model.
        """
        if hasattr(self.args, "aggregate_seq") and self.args.aggregate_seq:
            agg_params_dict = self.end_agg_seq(args)
            total_weights_delta = copy.deepcopy(agg_params_dict[MLMessage.MODEL_PARAMS]["agg_params"])
            total_c_delta_para = copy.deepcopy(agg_params_dict["c_delta_para"]["agg_params"])
        else:
            total_weights_delta = FedMLAggOperator.agg(self.args, raw_client_model_or_grad_list)
            key_op_list = [("c_delta_para", "avg")]
            agg_params_dict = self.sync_agg_params(self.client_result_dict, self.sample_num_dict, key_op_list)
            total_c_delta_para = agg_params_dict["c_delta_para"]["agg_params"]

        w_global = self.model.state_dict()
        for key in w_global.keys():
            w_global[key] += total_weights_delta[key] * self.args.server_lr
        # c_global_para = self.c_model_global.state_dict()
        for key in self.c_model_global.keys():
            if self.c_model_global[key].type() == 'torch.LongTensor':
                self.c_model_global[key] += total_c_delta_para[key].type(torch.LongTensor)
            elif self.c_model_global[key].type() == 'torch.cuda.LongTensor':
                self.c_model_global[key] += total_c_delta_para[key].type(torch.cuda.LongTensor)
            else:
                self.c_model_global[key] += total_c_delta_para[key]

        return w_global


    def before_agg(self, client_result_dict, sample_num_dict):
        self.client_result_dict = client_result_dict
        self.sample_num_dict = sample_num_dict


    def end_agg(self) -> Dict:
        other_result = dict()
        other_result["c_model_global"] = copy.deepcopy(self.c_model_global)
        self.initialize_params_dict()
        return other_result




    






