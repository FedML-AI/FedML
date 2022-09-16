import logging
from typing import List, Dict, Any

from .leave_one_out import LeaveOneOut


class ContributionAssessorManager:
    def __init__(self, args):
        self.args = args
        self.client_num_per_round = args.client_num_per_round
        self.assessor = self._build_assesor()

    def _build_assesor(self):
        if not hasattr(self.args, "contribution_alg"):
            return None
        if self.args.contribution_alg == "LOO":
            assessor = LeaveOneOut()
        else:
            raise Exception("no such algorithm for ContributionAssessor.")
        return assessor

    def run(
        self,
        model_list_from_client_update: List[Dict],
        model_aggregated: Dict,
        model_last_round: Dict,
        acc_on_aggregated_model: float,
        val_dataloader: Any,
    ) -> List[float]:
        if self.assessor is None:
            return None

        contribution_vector = self.assessor.run(
            self.client_num_per_round,
            model_list_from_client_update,
            model_aggregated,
            model_last_round,
            acc_on_aggregated_model,
            val_dataloader,
        )
        logging.info("ContributionAssessorManager.run() contribution_vector = {}".format(contribution_vector))
        return contribution_vector
