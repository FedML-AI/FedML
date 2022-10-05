import logging
from copy import deepcopy
from typing import List, Tuple, Dict
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.optim.optimizer import Optimizer, required

from ...core.common.ml_engine_backend import MLEngineBackend
from .base_client_optimizer import ClientOptimizer


from fedml.ml.ml_message import MLMessage



"""
FedNova Optimizer implementation cited from https://github.com/JYWa/FedNova/tree/master
"""


class FedNova(Optimizer):
    r"""Implements federated normalized averaging (FedNova).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        ratio (float): relative sample size of client
        gmf (float): global/server/slow momentum factor
        mu (float): parameter for proximal local SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(
        self,
        params,
        ratio,
        gmf,
        mu=0,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        variance=0,
    ):

        self.gmf = gmf
        self.ratio = ratio
        self.momentum = momentum
        self.mu = mu
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            variance=variance,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedNova, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FedNova, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        # scale = 1**self.itr
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                if "old_init" not in param_state:
                    param_state["old_init"] = torch.clone(p.data).detach()

                local_lr = group["lr"]

                # apply momentum updates
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply proximal updates
                if self.mu != 0:
                    d_p.add_(self.mu, p.data - param_state["old_init"])

                # update accumalated local updates
                if "cum_grad" not in param_state:
                    param_state["cum_grad"] = torch.clone(d_p).detach()
                    param_state["cum_grad"].mul_(local_lr)

                else:
                    param_state["cum_grad"].add_(local_lr, d_p)

                p.data.add_(-local_lr, d_p)

        # compute local normalizing vector a_i
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        self.etamu = local_lr * self.mu
        if self.etamu != 0:
            self.local_normalizing_vec *= 1 - self.etamu
            self.local_normalizing_vec += 1

        if self.momentum == 0 and self.etamu == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1

        return loss


class FedNovaClientOptimizer(ClientOptimizer):


    def preprocess(self, args, client_index, model, train_data, device, server_result, criterion):
        """
        1. Return params_to_update for update usage.
        2. pass model, train_data here, in case the algorithm need some preprocessing
        """

        params_to_client_optimizer = server_result[MLMessage.PARAMS_TO_CLIENT_OPTIMIZER]
        sample_num_dict = server_result[MLMessage.SAMPLE_NUM_DICT]
        round_sample_num = sum(list(sample_num_dict.values()))

        ratio = torch.FloatTensor(
                [sample_num_dict[client_index] / round_sample_num]
            ).to(device)

        self.optimizer = FedNova(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.learning_rate,
            gmf=self.args.gmf,
            mu=self.args.mu,
            ratio=ratio,
            momentum=self.args.momentum,
            dampening=self.args.dampening,
            weight_decay=self.args.weight_decay,
            nesterov=self.args.nesterov,
        )
        self.init_params = deepcopy(model.state_dict())
        return model

    def backward(self, args, client_index, model, x, labels, criterion, device, loss):
        """
        """
        loss.backward()
        return loss

    def update(self, args, client_index, model, x, labels, criterion, device):
        """
        """
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        self.optimizer.step()


    def end_local_training(self, args, client_index, model, train_data, device) -> Dict:
        """
        1. Return weights_or_grads, params_to_server_optimizer for special server optimizer need.
        """
        norm_grad = self.get_local_norm_grad(self.optimizer, model.state_dict(), self.init_params)
        tau_eff = self.get_local_tau_eff(self.optimizer)
        params_to_server_optimizer = {}
        params_to_server_optimizer["tau_eff"] = tau_eff
        return norm_grad, params_to_server_optimizer




    def get_local_norm_grad(self, opt, cur_params, init_params, weight=0):
        if weight == 0:
            weight = opt.ratio
        grad_dict = {}
        for k in cur_params.keys():
            scale = 1.0 / opt.local_normalizing_vec
            cum_grad = init_params[k] - cur_params[k]
            cum_grad.mul_(weight * scale)
            grad_dict[k] = cum_grad
        return grad_dict

    def get_local_tau_eff(self, opt):
        if opt.mu != 0:
            return opt.local_steps * opt.ratio
        else:
            return opt.local_normalizing_vec * opt.ratio











