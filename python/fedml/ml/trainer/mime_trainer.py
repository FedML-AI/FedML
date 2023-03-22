import copy
import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer

from fedml.simulation.sp.fedopt.optrepo import OptRepo
from fedml.simulation.sp.mime.opt_utils import OptimizerLoader

from fedml.utils.model_utils import get_named_data

import logging


def clip_norm(tensors, device, max_norm=1.0, norm_type=2.):
    total_norm = torch.norm(torch.stack(
        [torch.norm(p.detach(), norm_type).to(device) for p in tensors]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in tensors:
        p.mul_(clip_coef_clamped.to(p.device))
    return total_norm


class MimeModelTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)


    def accumulate_data_grad(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()
        model.zero_grad()

        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102

        batch_loss = []
        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            log_probs = model(x)
            loss = criterion(log_probs, labels)  # pylint: disable=E1102
            loss.backward()

            batch_loss.append(loss.item())
        logging.info(
            "Obtaining whole grad, Client Index = {}\t \tLoss: {:.6f}".format(
                self.id, sum(batch_loss) / len(batch_loss)
            )
        )
        local_grad = {}
        local_grad = get_named_data(model, mode="GRAD", use_cuda=False)
        return local_grad


    def train(self, train_data, device, args, grad_global, global_named_states):
        model = self.model

        model.to(device)
        model.train()

        init_model = copy.deepcopy(model)
        if not self.args.mimelite:
            init_model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )
        opt_loader = OptimizerLoader(model, optimizer)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()

                if not self.args.mimelite:
                    init_model.zero_grad()
                    log_probs = init_model(x)
                    loss_init_model = criterion(log_probs, labels)  # pylint: disable=E1102
                    loss_init_model.backward()

                    init_grad = {}
                    for name, parameter in init_model.named_parameters():
                        init_grad[name] = parameter.grad

                    for name, parameter in model.named_parameters():
                        parameter.grad = parameter.grad - init_grad[name] + grad_global[name].to(device)

                opt_loader.set_opt_state(copy.deepcopy(global_named_states), device)
                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )
        # local_grad = {}
        # init_model_params = init_model.state_dict()
        # for name, param in model.named_parameters():
        #     # local_grad[name] = torch.clamp(param.data - init_model_params[name], -1.0, 1.0)
        #     # local_grad[name] = torch.clamp(param.data - init_model_params[name], -0.1, 0.1)
        #     local_grad[name] = param.data - init_model_params[name]
        local_grad = self.accumulate_data_grad(train_data, device, args)

        clip_norm(list(local_grad.values()), device, max_norm=1.0, norm_type=2.)
        return local_grad


    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics
