import torch
import logging
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import copy

from ...core.common.ml_engine_backend import MLEngineBackend
from .base_client_optimizer import ClientOptimizer

from fedml.utils.model_utils import check_device

from fedml.ml.ml_message import MLMessage


class ScaffoldClientOptimizer(ClientOptimizer):


    def load_status(self, args, client_status):
        """
        Load status of client optimizer.
        """
        self.client_status = client_status


    def add_status(self, client_status) -> dict:
        """
        Load status of client optimizer.
        """
        client_status["c_model_local"] = self.c_model_local
        return client_status


    def preprocess(self, args, client_index, model, train_data, device, server_result, criterion):
        """
        1. Return params_to_update for update usage.
        2. pass model, train_data here, in case the algorithm need some preprocessing
        """

        params_to_client_optimizer = server_result[MLMessage.PARAMS_TO_CLIENT_OPTIMIZER]
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
        if "c_model_local" not in self.client_status:
            self.c_model_local = {}
            for name, params in model.named_parameters():
                self.c_model_local[name] = (params.data*0).cpu()
        else:
            self.c_model_local = self.client_status["c_model_local"]
        self.c_model_global = params_to_client_optimizer["c_model_global"]
        self.w_global = copy.deepcopy(model.state_dict())
        self.iteration_cnt = 0
        return model

    def backward(self, args, client_index, model, x, labels, criterion, device, loss):
        """
        """
        loss.backward()
        return loss


    def update(self, args, client_index, model, x, labels, criterion, device) -> Dict:
        """
        """
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.iteration_cnt += 1
        self.optimizer.step()
        current_lr = self.args.learning_rate
        for name, param in model.named_parameters():
            # logging.debug(f"c_model_global[name].device : {c_model_global[name].device}, \
            #     c_model_global_params[name].device : {c_model_local_params[name].device}")
            param.data = param.data - current_lr * \
                check_device((self.c_model_global[name] - self.c_model_local[name]), param.data.device)


    def end_local_training(self, args, client_index, model, train_data, device) -> Dict:
        """
        1. Return weights_or_grads, params_to_server_optimizer for special server optimizer need.
        """
        c_delta_para = {}
        global_model_para = self.w_global
        net_para = model.cpu().state_dict()
        weights_delta = {}
        c_new_para = {}

        # self.c_model_local

        for key in net_para:
            c_new_para[key] = self.c_model_local[key].cpu() - self.c_model_global[key].cpu() + \
                (global_model_para[key].cpu() - net_para[key]) / (self.iteration_cnt * self.args.learning_rate)
            c_delta_para[key] = c_new_para[key] - self.c_model_local[key].cpu()
            weights_delta[key] = net_para[key] - global_model_para[key].cpu()

        params_to_server_optimizer = {}
        params_to_server_optimizer["c_delta_para"] = c_delta_para
        return weights_delta, params_to_server_optimizer














