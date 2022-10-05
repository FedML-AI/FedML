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


    def load_status(self, args, client_status):
        """
        Load status of client optimizer.
        """
        self.client_status = client_status


    def add_status(self, client_status) -> dict:
        """
        Load status of client optimizer.
        """
        return client_status

    @abstractmethod
    def preprocess(self, args, client_index, model, train_data, device, server_result, criterion):
        """
        1. Return params_to_update for update usage.
        2. pass model, train_data here, in case the algorithm need some preprocessing
        """
        pass

    @abstractmethod
    def backward(self, args, client_index, model, x, labels, criterion, device, loss):
        """
        """
        pass


    @abstractmethod
    def update(self, args, client_index, model, x, labels, criterion, device):
        """
        """
        pass


    def end_local_training(self, args, client_index, model, train_data, device):
        """
        1. Return weights_or_grads, params_to_server_optimizer for special server optimizer need.
        """
        pass















