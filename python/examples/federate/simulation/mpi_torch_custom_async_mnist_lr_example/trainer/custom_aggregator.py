import logging

import torch
import torch.nn as nn

from fedml.core import ServerAggregator


class CustomAggregator(ServerAggregator):

    def __init__(self, model, args=None):
        super().__init__(model, args)

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        self.model.to(device)
        self.model.eval()

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
                pred = self.model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102
                # _, predicted = torch.max(pred, 1)
                # correct = predicted.eq(target).sum()

                # metrics["test_correct"] += correct.item()

        return metrics

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return True
