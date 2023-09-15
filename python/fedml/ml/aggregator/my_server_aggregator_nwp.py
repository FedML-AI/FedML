import copy
import logging

import torch
import wandb
from torch import nn

from ... import mlops
from ...core.alg_frame.server_aggregator import ServerAggregator


class MyServerAggregatorNWP(ServerAggregator):
    def get_model_params(self):
        """
        Get the model parameters.

        Returns:
            OrderedDict: The model parameters.
        """
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        """
        Set the model parameters.

        Args:
            model_parameters (OrderedDict): The model parameters to set.
        """
        self.model.load_state_dict(model_parameters)

    def _test(self, test_data, device, args):
        """
        Internal method for testing the model on a given dataset.

        Args:
            test_data: The test dataset.
            device: The device to run the test on.
            args: A dictionary containing configuration parameters.

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

    def test(self, test_data, device, args):
        """
        Test the model on a given dataset, log the results, and return test accuracy and loss.

        Args:
            test_data: The test dataset.
            device: The device to run the test on.
            args: A dictionary containing configuration parameters.

        Returns:
            tuple: A tuple containing test accuracy and loss.
        """
        # test data
        test_num_samples = []
        test_tot_corrects = []
        test_losses = []

        metrics = self._test(test_data, device, args)

        test_tot_correct, test_num_sample, test_loss = (
            metrics["test_correct"],
            metrics["test_total"],
            metrics["test_loss"],
        )
        test_tot_corrects.append(copy.deepcopy(test_tot_correct))
        test_num_samples.append(copy.deepcopy(test_num_sample))
        test_losses.append(copy.deepcopy(test_loss))

        # test on test dataset
        test_acc = sum(test_tot_corrects) / sum(test_num_samples)
        test_loss = sum(test_losses) / sum(test_num_samples)
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": args.round_idx})
            wandb.log({"Test/Loss": test_loss, "round": args.round_idx})

        mlops.log({"Test/Acc": test_acc, "round": args.round_idx})
        mlops.log({"Test/Loss": test_loss, "round": args.round_idx})

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        logging.info(stats)

        return (test_acc, test_loss, None, None)
