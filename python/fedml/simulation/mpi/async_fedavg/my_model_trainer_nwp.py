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

        # train and update
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
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
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

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

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, 1)
                target_pos = ~(target == 0)
                correct = (predicted.eq(target) * target_pos).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target_pos.sum().item()
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
