import logging
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from ...core.common.ml_engine_backend import MLEngineBackend

from .agg_operator import FedMLAggOperator



class ServerOptimizer(ABC):
    """Abstract base class for federated learning trainer.
    1. The goal of this abstract class is to be compatible to
    any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
    2. This class is used only on server side.
    3. This class is an operator which can caches states across rounds. 
    Because the server consistently exists.
    """
    def __init__(self, args):
        self.args = args
        self.initialize_params_dict()

    def initialize_params_dict(self):
        self.client_index_list = []
        self.params_to_server_optimizer_dict = dict()


    @abstractmethod
    def initialize(self, args, model):
        pass

    @abstractmethod
    def get_init_params(self) -> Dict:
        """
        1. Return init params_to_client_optimizer for special aggregator need.
        """
        pass
        # params_to_client_optimizer = dict()
        # return params_to_client_optimizer


    def add_params_to_server_optimizer(self, index, params_to_server_optimizer,):
        self.client_index_list.append(index)
        self.params_to_server_optimizer_dict[index] = params_to_server_optimizer



    def seq_agg_params(self):
        pass


    def sync_agg_params(self, sample_num_dict, key_op_list):
        # for i in range(len(sample_num_dict)):
        training_num = 0
        for client_index in self.client_index_list:
            local_sample_num = sample_num_dict[client_index]
            training_num += local_sample_num

        agg_params_dict = {}
        for key, op, in key_op_list:
            params_list = []
            for client_index in self.client_index_list:
                params_list.append((sample_num_dict[client_index], 
                    self.params_to_server_optimizer_dict[client_index][key]))

            agg_params = FedMLAggOperator.agg_with_weight(self.args, params_list, training_num, op)
            agg_params_dict[key] = agg_params
        return agg_params_dict


    @abstractmethod
    def agg(self, args, raw_client_model_or_grad_list):
        """
        Use this function to obtain the final global model.
        """
        pass


    @abstractmethod
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











