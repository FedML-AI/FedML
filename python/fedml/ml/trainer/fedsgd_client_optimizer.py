import torch
import logging
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from ...core.common.ml_engine_backend import MLEngineBackend
from .base_client_optimizer import ClientOptimizer


from fedml.ml.ml_message import MLMessage

from fedml.utils.model_utils import get_all_bn_params, get_named_data

# from fedml.core.compression import MLcompression


class FedSGDClientOptimizer(ClientOptimizer):
    def load_status(self, args, client_status):
        """
        Load status of client optimizer.
        """
        self.client_status = client_status


    def add_status(self, client_status):
        return client_status


    def preprocess(self, args, client_index, model, train_data, device, model_optimizer, criterion):
        server_weights = self.server_result.get(MLMessage.MODEL_PARAMS)
        model.load_state_dict(server_weights)
        return model


    def backward(self, args, client_index, model, x, labels, criterion, device, loss):
        """
        """
        loss.backward()
        return loss


    def update(self, args, client_index, model, x, labels, criterion, device):
        """
            SGD return grad to the server, not update at client side
        """
        pass


    def end_local_training(self, args, client_index, model, train_data, device):
        other_result = dict()
        named_grads = get_named_data(model, mode='GRAD', use_cuda=False)
        other_result[MLMessage.MODEL_PARAMS] = named_grads
        bn_params = get_all_bn_params(model)
        other_result["bn_params"] = bn_params
        return other_result

        # return model.cpu().state_dict(), {}














