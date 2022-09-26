import torch
import logging
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

from ...core.common.ml_engine_backend import MLEngineBackend
from .base_train_operator import ClientOperator




class FedAvgClientOperator(ClientOperator):

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
        return model

    def backward(self, args, client_id, model, train_data, device, loss, params_to_operator):
        """
        """
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














