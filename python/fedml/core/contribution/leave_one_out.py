import logging
import numpy as np
from typing import List, Dict, Callable, Any

from .base_contribution_assessor import BaseContributionAssessor


class LeaveOneOut(BaseContributionAssessor):
    def run(
        self,
        num_client_for_this_round: int,
        model_list_from_client_update: List[Dict],
        model_aggregated: Dict,
        model_last_round: Dict,
        acc_on_aggregated_model: float,
        val_dataloader: Any,
        validation_func: Callable[[Dict, Any, Any], float],
        device,
    ) -> List[float]:
        # accuracy of the aggregated model
        acc_on_aggregated_model = validation_func(model_aggregated, val_dataloader, device)
        contributions = np.zeros(num_client_for_this_round, dtype='f')
        for client in range(num_client_for_this_round):
            # assuming same number of samples in each client
            model_aggregated_wo_client = np.sum(i for i in model_list_from_client_update if i != client) / (num_client_for_this_round-1)
            acc_wo_client = validation_func(model_aggregated_wo_client, val_dataloader, device)
            contributions[client] = acc_on_aggregated_model-acc_wo_client
        logging.info("contributions = {}".format(contributions))
        return contributions
