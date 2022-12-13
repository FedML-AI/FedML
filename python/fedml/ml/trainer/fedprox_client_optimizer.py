import torch
import logging
from copy import deepcopy
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from ...core.common.ml_engine_backend import MLEngineBackend
from .base_client_optimizer import ClientOptimizer


from fedml.ml.ml_message import MLMessage




class FedProxClientOptimizer(ClientOptimizer):


    def preprocess(self, args, client_index, model, train_data, device, model_optimizer, criterion):
        self.model_optimizer = model_optimizer
        self.global_model_params = deepcopy(model.state_dict())
        return model


    def backward(self, args, client_index, model, x, labels, criterion, device, loss):
        fed_prox_reg = 0.0
        for name, param in model.named_parameters():
            fed_prox_reg += ((args.fedprox_mu / 2) * \
                torch.norm((param - self.global_model_params[name].data.to(device)))**2)
        loss += fed_prox_reg
        loss.backward()
        return loss

    def update(self, args, client_index, model, x, labels, criterion, device) -> Dict:
        """
        """
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        self.model_optimizer.step()

    def end_local_training(self, args, client_index, model, train_data, device) -> Dict:
        # return model.cpu().state_dict(), {}
        other_result = dict()
        other_result[MLMessage.MODEL_PARAMS] = model.cpu().state_dict()
        return other_result














