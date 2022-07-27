from typing import List, Tuple, Dict

import torch
from torch import nn

from .agg_operator import FedMLAggOperator
from ...core.alg_frame.server_aggregator import ServerAggregator


class MyServerAggregatorNWP(ServerAggregator):
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

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, 1)
                target_pos = ~(target == 0)
                correct = (predicted.eq(target) * target_pos).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target_pos.sum().item()
        return metrics
