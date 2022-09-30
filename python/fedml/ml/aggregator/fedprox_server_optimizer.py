import logging
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from ...core.common.ml_engine_backend import MLEngineBackend

from .agg_operator import FedMLAggOperator

class FedProxServerOptimizer(ABC):
    """Abstract base class for federated learning trainer.
    1. The goal of this abstract class is to be compatible to
    any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
    2. This class is used only on server side.
    3. This class is an operator which can caches states across rounds. 
    Because the server consistently exists.
    """
    def __init__(self, args):
        self.args = args
        self.params_to_server_optimizer_dict = dict()

    @abstractmethod
    def initialize(self, args, model):
        pass


    def get_init_params(self) -> Dict:
        """
        1. Return init params_to_client_optimizer for special aggregator need.
        """
        params_to_client_optimizer = dict()
        return params_to_client_optimizer


    def add_params_to_server_optimizer(self, index, params_to_server_optimizer,):
        self.params_to_server_optimizer_dict[index] = params_to_server_optimizer

    @abstractmethod
    def agg(self, args, raw_client_model_or_grad_list):
        """
        Use this function to obtain the final global model.
        """
        pass


    @abstractmethod
    def before_agg(self):
        pass

    def end_agg(self) -> Dict:
        """
        1. Return params_to_client_optimizer for special aggregator need.
        """
        params_to_client_optimizer = dict()
        return params_to_client_optimizer











