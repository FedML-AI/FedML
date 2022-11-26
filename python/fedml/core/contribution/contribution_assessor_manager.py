import logging
from typing import List, Dict, Any, Callable

from .gtg_shapley_value import GTGShapleyValue
from .leave_one_out import LeaveOneOut
from .mr_shapley_value import MRShapleyValue


class ContributionAssessorManager:
    def __init__(self, args):
        self.args = args
        self.assessor = self._build_assesor()

    def _build_assesor(self):
        if not hasattr(self.args, "contribution_alg"):
            return None
        if self.args.contribution_alg == "LOO":
            assessor = LeaveOneOut(self.args)
        elif self.args.contribution_alg == "GTG":
            assessor = GTGShapleyValue(self.args)
        elif self.args.contribution_alg == "MR":
            assessor = MRShapleyValue(self.args)
        else:
            raise Exception("no such algorithm for ContributionAssessor.")
        return assessor

    def run(
        self,
        client_num_per_round,
        client_index_for_this_round,
        aggregation_func: Callable,
        local_weights_from_clients: List[Dict],
        acc_on_last_round: float,
        acc_on_aggregated_model: float,
        val_dataloader: Any,
        validation_func: Callable[[Dict, Any, Any], float],
        device,
    ):
        self.assessor.run(
            client_num_per_round,
            client_index_for_this_round,
            aggregation_func,
            local_weights_from_clients,
            acc_on_last_round,
            acc_on_aggregated_model,
            val_dataloader,
            validation_func,
            device,
        )

    def get_final_contribution_assignment(self):
        return self.assessor.get_final_contribution_assignment()
