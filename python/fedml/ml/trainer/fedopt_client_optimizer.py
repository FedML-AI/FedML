import torch
import logging
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from ...core.common.ml_engine_backend import MLEngineBackend
from .base_client_optimizer import ClientOptimizer


from fedml.ml.ml_message import MLMessage


class FedOptClientOptimizer(ClientOptimizer):

    def preprocess(self, args, client_index, model, train_data, device, model_optimizer, criterion):
        self.model_optimizer = model_optimizer
        return model

    def backward(self, args, client_index, model, x, labels, criterion, device, loss):
        """
        """
        loss.backward()
        return loss


    def update(self, args, client_index, model, x, labels, criterion, device):
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.model_optimizer.step()


    def end_local_training(self, args, client_index, model, train_data, device) -> Dict:
        other_result = dict()
        other_result[MLMessage.MODEL_PARAMS] = model.cpu().state_dict()
        return other_result

        # return model.cpu().state_dict(), {}














