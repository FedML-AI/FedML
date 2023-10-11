import torch
from torch import nn
from ....core.alg_frame.client_trainer import ClientTrainer
import logging

class MyModelTrainer(ClientTrainer):
    """
    Custom client trainer for federated learning using PyTorch.

    Methods:
        get_model_params: Get the model parameters as a state dictionary.
        set_model_params: Set the model parameters from a state dictionary.
        train: Train the model on the given training data.
        test: Evaluate the model on the given test data.
        test_on_the_server: Perform server-side testing (not implemented).

    Parameters:
        model: The PyTorch model to be trained.
        id (int): The identifier of the client.
    """

    def get_model_params(self):
        """
        Get the model parameters as a state dictionary.

        Returns:
            dict: Model parameters as a state dictionary.
        """
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        """
        Set the model parameters from a state dictionary.

        Args:
            model_parameters (dict): Model parameters as a state dictionary.
        """
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, lr=None):
        """
        Train the model on the given training data.

        Args:
            train_data: Training data for the client.
            device: Device (e.g., GPU or CPU) for model training.
            args: Command-line arguments for training configuration.
            lr (float): Learning rate for optimization (optional).

        Returns:
            None
        """
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                momentum=args.momentum
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
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
                loss.backward()

                # Uncomment this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                logging.info(
                    "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * args.batch_size,
                        len(train_data) * args.batch_size,
                        100.0 * (batch_idx + 1) / len(train_data),
                        loss.item(),
                    )
                )
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def test(self, test_data, device, args):
        """
        Evaluate the model on the given test data.

        Args:
            test_data: Test data for the client.
            device: Device (e.g., GPU or CPU) for model evaluation.
            args: Command-line arguments for evaluation configuration.

        Returns:
            dict: Evaluation metrics, including test_correct, test_loss, and test_total.
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

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        """
        Perform server-side testing (not implemented).

        Args:
            train_data_local_dict: Local training data for all clients.
            test_data_local_dict: Local test data for all clients.
            device: Device (e.g., GPU or CPU) for testing.
            args: Command-line arguments for testing configuration (not used).

        Returns:
            bool: Always returns False (not implemented).
        """
        return False
