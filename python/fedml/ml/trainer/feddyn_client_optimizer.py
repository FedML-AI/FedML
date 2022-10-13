import torch
import logging
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import copy

from ...core.common.ml_engine_backend import MLEngineBackend
from .base_client_optimizer import ClientOptimizer

from fedml.utils.model_utils import check_device
from fedml.utils.model_utils import named_params_to

from fedml.ml.ml_message import MLMessage


class FedDynClientOptimizer(ClientOptimizer):


    def load_status(self, args, client_status):
        """
        Load status of client optimizer.
        """
        self.client_status = client_status


    def add_status(self, client_status) -> dict:
        """
        Load status of client optimizer.
        """
        client_status["old_grad"] = self.old_grad
        return client_status


    def preprocess(self, args, client_index, model, train_data, device, server_result, criterion):
        # params_to_client_optimizer = server_result[MLMessage.PARAMS_TO_CLIENT_OPTIMIZER]
        if args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )
        if "old_grad" not in self.client_status:
            self.old_grad = {}
            for name, params in model.named_parameters():
                self.old_grad[name] = (params.data*0).cpu()
        else:
            self.old_grad = self.client_status["old_grad"]
        self.old_grad = named_params_to(self.old_grad, device)
        self.global_model_params = copy.deepcopy(model.state_dict())
        self.iteration_cnt = 0
        return model

    def backward(self, args, client_index, model, x, labels, criterion, device, loss):
        """
        """
        #=== Dynamic regularization === #
        lin_penalty = 0.0
        norm_penalty = 0.0
        for name, param in model.named_parameters():
            # Linear penalty
            # lin_penalty += torch.sum(param.data * old_grad[name])
            lin_penalty += (self.args.feddyn_alpha / 2) * torch.sum(param.data * self.old_grad[name]) 
            # Quadratic Penalty
            # norm_penalty += (self.args.feddyn_alpha / 2) * torch.norm((param.data - self.global_model_params[name].data.to(device)))**2
            norm_penalty += (self.args.feddyn_alpha / 2) * torch.norm((param.data - self.global_model_params[name].data))**2
        loss = loss - lin_penalty + norm_penalty
        self.optimizer.zero_grad()

        loss.backward()
        return loss


    def update(self, args, client_index, model, x, labels, criterion, device) -> Dict:
        """
        """
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()


    def end_local_training(self, args, client_index, model, train_data, device) -> Dict:

        for name, param in model.named_parameters():
            self.old_grad[name] = (self.old_grad[name] - self.args.feddyn_alpha * (
                param.data - self.global_model_params[name]))

        self.old_grad = named_params_to(self.old_grad, "cpu")

        other_result = dict()
        other_result[MLMessage.MODEL_PARAMS] = model.cpu().state_dict()
        return other_result

        # return model.cpu().state_dict(), {}














