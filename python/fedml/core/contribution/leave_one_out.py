import logging
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
        acc = validation_func(model_aggregated, val_dataloader, device)
        logging.info("acc = {}".format(acc))
        return [i*0.1 for i in range(num_client_for_this_round)]
