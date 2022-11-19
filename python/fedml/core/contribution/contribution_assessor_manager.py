import logging
from typing import List, Dict, Any, Callable

from .leave_one_out import LeaveOneOut
from .GTG-Shapley import GTG_Shapley
from .Exact_SV import Exact_SV

class ContributionAssessorManager:
    def __init__(self, args):
        self.args = args
        self.client_num_per_round = args.client_num_per_round
        # TO DO: we need to add the indices of the participating clients here.
        self.assessor = self._build_assesor()

    def _build_assesor(self):
        if not hasattr(self.args, "contribution_alg"):
            return None
        if self.args.contribution_alg == "LOO":
            assessor = LeaveOneOut()
        if self.args.contribution_alg == "GTG":
            assessor = GTG_Shapley()
        if self.args.contribution_alg == "ExactSV":
            assessor = Exact_SV()
        else:
            raise Exception("no such algorithm for ContributionAssessor.")
        return assessor

    def run(
        self,
        fraction: Dict,  # this is the weights of the clients in FedAvg
        local_weights_from_clients: List[Dict],
        model_aggregated: Dict,
        model_last_round: Dict,
        acc_on_aggregated_model: float,
        val_dataloader: Any,
        validation_func: Callable[[Dict, Any, Any], float],
        device,
    ) -> List[float]:
        if self.assessor is None:
            return None

        contribution_vector = self.assessor.run(
            self.client_num_per_round,
            # TO DO: we need to add the indices of the participating clients here.
            fraction,  # this is the weights of the clients in FedAvg
            local_weights_from_clients,
            model_aggregated,
            model_last_round,
            acc_on_aggregated_model,
            val_dataloader,
            validation_func,
            device,
        )
        logging.info("ContributionAssessorManager.run() contribution_vector = {}".format(contribution_vector))
        return contribution_vector
