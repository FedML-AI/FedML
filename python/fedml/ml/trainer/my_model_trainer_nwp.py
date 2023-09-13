import time
import logging

import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer


class ModelTrainerNWP(ClientTrainer):
    """
    A custom model trainer for Next Word Prediction (NWP) tasks.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        args: Training arguments.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be trained.
        args: Training arguments.

    Methods:
        get_model_params(): Get the model parameters as a state dictionary.
        set_model_params(model_parameters): Set the model parameters from a state dictionary.
        train(train_data, device, args): Train the model.
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

    def train(self, train_data, device, args):
        """
        Train the model.

        Args:
            train_data: The training data.
            device (torch.device): The device (CPU or GPU) to use for training.
            args: Training arguments.
        """
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(
            device)  # pylint: disable=E1102
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
            # begin_time = time.time()
            # current_time = time.time()
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)

                # logging.info(f"data loading time: {time.time() - current_time}")
                # current_time = time.time()
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                optimizer.step()
                # logging.info(f"backward and update time: {time.time() - begin_time}")
                # current_time = time.time()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args):
        """
        Evaluate the model on test data and return evaluation metrics.

        Args:
            test_data: The test data.
            device (torch.device): The device (CPU or GPU) to use for evaluation.
            args: Training arguments.

        Returns:
            dict: Evaluation metrics including test accuracy, test loss, and the total number of test samples.
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
