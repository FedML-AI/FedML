import copy
import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer
import logging



import torch

def model_parameter_vector(model):
    """
    Flatten the parameters of a PyTorch model into a single 1D tensor.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        torch.Tensor: A 1D tensor containing all the flattened model parameters.
    """
    param = [p.view(-1) for p in model.parameters()]
    return torch.cat(param, dim=0)  # Use torch.cat to concatenate tensors

def parameter_vector(parameters):
    """
    Flatten a dictionary of PyTorch parameters into a single 1D tensor.

    Args:
        parameters (dict): A dictionary of PyTorch parameters.

    Returns:
        torch.Tensor: A 1D tensor containing all the flattened parameters.
    """
    param = [p.view(-1) for p in parameters.values()]
    return torch.cat(param, dim=0)  # Use torch.cat to concatenate tensors



class FedDynModelTrainer(ClientTrainer):
    """
    A federated dynamic model trainer that implements training and testing methods.

    Args:
        ClientTrainer: The base class for client trainers.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be trained.
        id (int): The identifier of the client.

    Methods:
        get_model_params(): Get the model parameters as a state dictionary.
        set_model_params(model_parameters): Set the model parameters from a state dictionary.
        train(train_data, device, args, old_grad): Train the model with federated dynamic regularization.
        test(test_data, device, args): Evaluate the model on test data and return evaluation metrics.
    """
    def get_model_params(self):
        """
        Get the model parameters as a state dictionary.

        Returns:
            dict: The model parameters as a state dictionary.
        """
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        """
        Set the model parameters from a state dictionary.

        Args:
            model_parameters (dict): The model parameters as a state dictionary.
        """
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, old_grad):
        """
        Train the model with federated dynamic regularization.

        Args:
            train_data: The training data.
            device (torch.device): The device (CPU or GPU) to use for training.
            args: Training arguments.
            old_grad (torch.Tensor): The previous gradient.

        Returns:
            torch.Tensor: The updated gradient.
        """
        model = self.model
        for params in model.parameters():
            params.requires_grad = True
        model.to(device)
        model.train()
        # old_grad = old_grad.to(device)
        flat_grad = parameter_vector(old_grad).to(device)
        # flat_grad = parameter_vector(old_grad).to(device).detach()

        global_model = copy.deepcopy(model)
        global_model_vector = model_parameter_vector(global_model)
        # global_model_vector = model_parameter_vector(global_model).detach()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay+self.args.feddyn_alpha,
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
                # lin_penalty = 0.0
                # norm_penalty = 0.0
                # for name, param in self.model.named_parameters():
                #     # Linear penalty
                #     lin_penalty += torch.sum(param.data * prev_grads[name])
                #     # Quadratic Penalty
                #     norm_penalty = (self.args.feddyn_alpha / 2) * torch.norm((param.data - previous_model[name].data.to(device)))**2
                # loss = loss - lin_penalty + norm_penalty
                # v1 = model_parameter_vector(model)
                # # https://github.com/alpemreacar/FedDyn/blob/48a19fac440ef079ce563da8e0c2896f8256fef9/utils_general.py#L219
                # lin_penalty = (self.args.feddyn_alpha  / 2 ) * torch.sum(v1 * flat_grad)
                # norm_penalty = (self.args.feddyn_alpha  / 2 ) * torch.norm(v1 - global_model_vector)
                # loss = loss - lin_penalty + norm_penalty


                v1 = model_parameter_vector(model)
                # https://github.com/alpemreacar/FedDyn/blob/48a19fac440ef079ce563da8e0c2896f8256fef9/utils_general.py#L219
                loss += self.args.feddyn_alpha * torch.sum(v1 * (- global_model_vector + flat_grad))


                # https://github.com/TsingZ0/PFL-Non-IID/blob/fd23a2124265fac69c137b313e66e45863487bd5/system/flcore/clients/clientdyn.py#L54
                # loss += self.args.feddyn_alpha/2 * torch.norm(v1 - global_model_vector, 2)
                # loss -= torch.dot(v1, old_grad)

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
        # https://github.com/alpemreacar/FedDyn/blob/48a19fac440ef079ce563da8e0c2896f8256fef9/utils_general.py#L219
        # old_grad += v1 - global_model_vector
        global_params = global_model.state_dict()
        current_params = model.state_dict()
        for key in global_params.keys():
            # old_grad[key] += (current_params[key] - global_params[key]).to(device)
            old_grad[key] += (current_params[key] - global_params[key]).to("cpu")

        # https://github.com/TsingZ0/PFL-Non-IID/blob/fd23a2124265fac69c137b313e66e45863487bd5/system/flcore/clients/clientdyn.py#L54
        # v1 = model_parameter_vector(model).detach()
        # old_grad = (old_grad - self.args.feddyn_alpha * (v1 - global_model_vector)).cpu()
        # old_grad = (old_grad + self.args.feddyn_alpha * (v1 - global_model_vector)).cpu()
        # Freeze model
        for params in model.parameters():
            params.requires_grad = False
            model.eval()
        return old_grad


    def test(self, test_data, device, args):
        """
        Evaluate the model on test data and return evaluation metrics.

        Args:
            test_data: The test data.
            device (torch.device): The device (CPU or GPU) to use for evaluation.
            args: Training arguments.

        Returns:
            dict: Evaluation metrics including test accuracy and test loss.
        """

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
