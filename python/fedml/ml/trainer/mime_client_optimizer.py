import torch
from torch import nn

import logging
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod
import copy

from ...core.common.ml_engine_backend import MLEngineBackend
from .base_client_optimizer import ClientOptimizer

from fedml.utils.model_utils import check_device

from fedml.utils.model_utils import get_named_data, clip_norm

from fedml.ml.utils.optrepo import OptRepo
from fedml.ml.utils.opt_utils import OptimizerLoader


from fedml.ml.ml_message import MLMessage


class MimeClientOptimizer(ClientOptimizer):


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



    def preprocess(self, args, client_index, model, train_data, device, server_result, criterion):
        self.client_index = client_index
        # params_to_client_optimizer = server_result[MLMessage.PARAMS_TO_CLIENT_OPTIMIZER]
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )
        self.optimizer = optimizer
        self.opt_loader = OptimizerLoader(model, self.optimizer)

        self.grad_global = server_result["grad_global"]
        self.global_named_states = server_result["global_named_states"]
        self.criterion = criterion

        self.init_model = copy.deepcopy(model)
        if not self.args.mimelite:
            self.init_model.train()
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

        if not self.args.mimelite:
            self.init_model.zero_grad()
            log_probs = self.init_model(x)
            loss_init_model = criterion(log_probs, labels)  # pylint: disable=E1102
            loss_init_model.backward()

            init_grad = {}
            for name, parameter in self.init_model.named_parameters():
                init_grad[name] = parameter.grad

            for name, parameter in model.named_parameters():
                parameter.grad = parameter.grad - init_grad[name] + self.grad_global[name].to(device)

        self.opt_loader.set_opt_state(copy.deepcopy(self.global_named_states), device)
        self.optimizer.step()


    def end_local_training(self, args, client_index, model, train_data, device) -> Dict:
        local_grad = self.accumulate_data_grad(train_data, model, device, args)

        clip_norm(list(local_grad.values()), device, max_norm=1.0, norm_type=2.)
        # params_to_server_optimizer = {}
        # params_to_server_optimizer["local_grad"] = local_grad
        # return model.cpu().state_dict(), params_to_server_optimizer
        other_result = dict()
        other_result[MLMessage.MODEL_PARAMS] = model.cpu().state_dict()
        other_result["local_grad"] = local_grad
        return other_result





    def accumulate_data_grad(self, train_data, model, device, args):

        batch_loss = []
        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            log_probs = model(x)
            loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
            loss.backward()

            batch_loss.append(loss.item())
        logging.info(
            "Obtaining whole grad, Client Index = {}\t \tLoss: {:.6f}".format(
                self.client_index, sum(batch_loss) / len(batch_loss)
            )
        )
        local_grad = {}
        local_grad = get_named_data(model, mode="GRAD", use_cuda=False)
        return local_grad









