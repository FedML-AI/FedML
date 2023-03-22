import logging
import copy

import torch
from torch import nn
from .fednova import FedNova


class Client:
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        args,
        device,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device

        """
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        """
        if self.args.dataset == "stackoverflow_lr":
            self.criterion = nn.BCELoss(reduction="sum").to(device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(device)

    def update_local_dataset(
        self, client_idx, local_training_data, local_test_data, local_sample_number
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

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

    def reset_fednova_optimizer(self, opt):
        opt.local_counter = 0
        opt.local_normalizing_vec = 0
        opt.local_steps = 0
        for group in opt.param_groups:
            for p in group["params"]:
                param_state = opt.state[p]
                param_state["cum_grad"].zero_()
                # Reinitialize momentum buffer
                if "momentum_buffer" in param_state:
                    param_state["momentum_buffer"].zero_()

    def train(self, net, ratio):
        net.train()
        # train and update
        init_params = copy.deepcopy(net.state_dict())
        optimizer = FedNova(
            net.parameters(),
            lr=self.args.learning_rate,
            gmf=self.args.gmf,
            mu=self.args.mu,
            ratio=ratio,
            momentum=self.args.momentum,
            dampening=self.args.dampening,
            weight_decay=self.args.wd,
            nesterov=self.args.nesterov,
        )

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.local_training_data):
                x, labels = x.to(self.device), labels.to(self.device)
                net = net.to(self.device)
                net.zero_grad()
                log_probs = net(x)
                loss = self.criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))
        norm_grad = self.get_local_norm_grad(optimizer, net.state_dict(), init_params)
        tau_eff = self.get_local_tau_eff(optimizer)
        # self.reset_fednova_optimizer(optimizer)
        return sum(epoch_loss) / len(epoch_loss), norm_grad, tau_eff

    def local_test(self, model_global, b_use_test_dataset=False):
        model_global.eval()
        model_global.to(self.device)
        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = model_global(x)
                loss = self.criterion(pred, target)  # pylint: disable=E1102

                if self.args.dataset == "stackoverflow_lr":
                    predicted = (pred > 0.5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > 0.1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics["test_precision"] += precision.sum().item()
                    metrics["test_recall"] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)

        return metrics
