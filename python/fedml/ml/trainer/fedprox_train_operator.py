import torch
import logging
from copy import deepcopy
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from ...core.common.ml_engine_backend import MLEngineBackend
from .base_train_operator import ClientOperator




class FedProxClientOperator(ClientOperator):

    def preprocess(self, args, model, train_data, device, params_to_operator) -> (model, Dict):
        """
        1. Return params_to_update for update usage.
        2. pass model, train_data here, in case the algorithm need some preprocessing
        """
        if args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )
        self.global_model_params = deepcopy(model.state_dict())
        return model


    def backward(self, args, client_id, model, train_data, device, loss, params_to_operator):
        # global_model_params = params_to_update["global_model"]
        fed_prox_reg = 0.0
        for name, param in model.named_parameters():
            fed_prox_reg += ((args.fedprox_mu / 2) * \
                torch.norm((param - self.global_model_params[name].data.to(device)))**2)
        loss += fed_prox_reg
        loss.backward()

    def update(self, args, model, train_data, device, params_to_operator) -> Dict:
        """
        """
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        self.optimizer.step()


    def end_local_training(self, args, model, train_data, device, params_to_operator) -> Dict:
        """
        1. Return params_to_agg for special aggregator need.
        """
        return None














