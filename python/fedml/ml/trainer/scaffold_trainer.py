import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer
from ...utils.model_utils import check_device
import logging


class ScaffoldModelTrainer(ClientTrainer):
    """
    A scaffold model trainer that implements training and testing methods.

    Args:
        ClientTrainer: The base class for client trainers.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be trained.
        id (int): The identifier of the client.

    Methods:
        get_model_params(): Get the model parameters as a state dictionary.
        set_model_params(model_parameters): Set the model parameters from a state dictionary.
        train(train_data, device, args, c_model_global_params, c_model_local_params): Train the model.
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

    def train(self, train_data, device, args, c_model_global_params, c_model_local_params):
        """
        Train the model.

        Args:
            train_data: The training data.
            device (torch.device): The device (CPU or GPU) to use for training.
            args: Training arguments.
            c_model_global_params (dict): Global model parameters.
            c_model_local_params (dict): Local model parameters.

        Returns:
            int: The number of training iterations.
        """
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        iteration_cnt = 0
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()

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
                current_lr = self.args.learning_rate
                for name, param in model.named_parameters():
                    # logging.debug(f"c_model_global[name].device : {c_model_global[name].device}, \
                    #     c_model_global_params[name].device : {c_model_local_params[name].device}")
                    param.data = param.data - current_lr * \
                        check_device(
                            (c_model_global_params[name] - c_model_local_params[name]), param.data.device)
                iteration_cnt += 1
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
        return iteration_cnt

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
