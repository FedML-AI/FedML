import logging

import torch
from torch import nn

from ....core.alg_frame.client_trainer import ClientTrainer


class MyModelTrainer(ClientTrainer):
    """
    Custom client model trainer for federated learning.

    Args:
        None

    Methods:
        get_model_params(): Get the model parameters.
        set_model_params(model_parameters): Set the model parameters.
        train(train_data, device, args): Train the model on the client.
        test(test_data, device, args): Test the model on the client.
        test_on_the_server(train_data_local_dict, test_data_local_dict, device, args=None): 
            Test the model on the server (not implemented in this class).
    """
    def get_model_params(self):
        """
        Get the model parameters.

        Returns:
            dict: The model parameters as a dictionary.
        """
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        """
        Set the model parameters.

        Args:
            model_parameters (dict): A dictionary containing the model parameters to set.
        """
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        """
        Train the model on the client.

        Args:
            train_data (torch.utils.data.DataLoader): DataLoader containing training data.
            device (torch.device): The device (CPU or GPU) to train on.
            args: Additional training arguments.

        Returns:
            None
        """
        model = self.model

        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.wd,
                amsgrad=True,
            )
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # logging.info(images.shape)
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    "(Trainer_ID {}. Local Training Epoch: {} \tLoss: {:.6f}".format(
                        self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                    )
                )

    def test(self, test_data, device, args):
        """
        Test the model on the client.

        Args:
            test_data (torch.utils.data.DataLoader): DataLoader containing test data.
            device (torch.device): The device (CPU or GPU) to test on.
            args: Additional testing arguments.

        Returns:
            dict: A dictionary containing test metrics.
        """
        model = self.model

        model.eval()
        model.to(device)

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }

        criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102
                if args.dataset == "stackoverflow_lr":
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

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        """
        Test the model on the server (not implemented in this class).

        Args:
            train_data_local_dict (dict): Dictionary containing local training data.
            test_data_local_dict (dict): Dictionary containing local test data.
            device (torch.device): The device (CPU or GPU) to test on.
            args: Additional testing arguments.

        Returns:
            bool: Always returns False in this implementation.
        """
        return False
