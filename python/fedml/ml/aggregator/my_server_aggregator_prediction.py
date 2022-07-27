from typing import List, Tuple, Dict

import torch
from torch import nn

from .agg_operator import FedMLAggOperator
from ...core.alg_frame.server_aggregator import ServerAggregator


class MyServerAggregatorTAGPred(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def aggregate(self, raw_client_model_or_grad_list: List[Tuple[float, Dict]]) -> Dict:
        return FedMLAggOperator.FedAVG(raw_client_model_or_grad_list)

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }

        """
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        """
        criterion = nn.BCELoss(reduction="sum").to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                predicted = (pred > 0.5).int()
                correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                true_positive = ((target * predicted) > 0.1).int().sum(axis=-1)
                precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                recall = true_positive / (target.sum(axis=-1) + 1e-13)
                metrics["test_precision"] += precision.sum().item()
                metrics["test_recall"] += recall.sum().item()
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)

        return metrics
