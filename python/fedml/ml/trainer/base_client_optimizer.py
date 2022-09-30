import logging
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from ...core.common.ml_engine_backend import MLEngineBackend





class ClientOptimizer(ABC):
    """Abstract base class for federated learning trainer.
    1. The goal of this abstract class is to be compatible to
    any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
    2. This class is used only on client side.
    3. This class is an operator which does not cache any states across rounds, but cache something during local training.
    """
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def preprocess(self, args, client_id, model, train_data, device, params_to_client_optimizer):
        """
        1. Return params_to_update for update usage.
        2. pass model, train_data here, in case the algorithm need some preprocessing
        """
        pass

    @abstractmethod
    def backward(self, args, client_id, model, train_data, device, loss, params_to_client_optimizer):
        """
        """
        pass


    @abstractmethod
    def update(self, args, client_id, model, train_data, device, params_to_client_optimizer) -> Dict:
        """
        """
        pass


    @abstractmethod
    def end_local_training(self, args, client_id, model, train_data, device, params_to_client_optimizer) -> Dict:
        """
        1. Return params_to_agg for special aggregator need.
        """
        pass














