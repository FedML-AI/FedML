import copy
import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer
import logging



def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.concat(param, dim=0)


def parameter_vector(parameters):
    param = [p.view(-1) for p in parameters.values()]
    return torch.concat(param, dim=0)


class FedDynModelTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, old_grad):
        model = self.model
        for params in model.parameters():
            params.requires_grad = True
        model.to(device)
        model.train()

        global_model = copy.deepcopy(model)
        global_model_params = global_model.state_dict()

        old_grad = {
            key: data.to(device) for key, data in old_grad.items()
        }

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                # weight_decay=args.weight_decay+self.args.feddyn_alpha,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay+self.args.feddyn_alpha,
                amsgrad=True,
            )


        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102

                #=== Dynamic regularization === #
                lin_penalty = 0.0
                norm_penalty = 0.0
                for name, param in model.named_parameters():
                    # Linear penalty
                    # lin_penalty += torch.sum(param.data * old_grad[name])
                    lin_penalty += (self.args.feddyn_alpha / 2) * torch.sum(param.data * old_grad[name]) 
                    # Quadratic Penalty
                    norm_penalty += (self.args.feddyn_alpha / 2) * torch.norm((param.data - global_model_params[name].data.to(device)))**2
                loss = loss - lin_penalty + norm_penalty

                optimizer.zero_grad()
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10.0) # Clip gradients
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients

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

        for name, param in model.named_parameters():
            old_grad[name] = (old_grad[name] - self.args.feddyn_alpha * (
                param.data - global_model_params[name])).to(device)

        for params in model.parameters():
            params.requires_grad = False
            model.eval()
        return old_grad


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
