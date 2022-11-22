from typing import List, Dict, Callable, Any

from scipy.special import comb

from .base_contribution_assessor import BaseContributionAssessor

class NaiveShapleyValue(BaseContributionAssessor):
    def __init__(self, args):
        super().__init__()
        self.SV = {}  # dict: {id:SV,...}

        self.args = args

    def run(
        self,
        num_client_for_this_round: int,
        client_index_for_this_round: List,
        aggregation_func: Callable,
        local_weights_from_clients: List[Dict],
        acc_on_last_round: float,
        acc_on_aggregated_model: float,
        val_dataloader: Any,
        validation_func: Callable[[Dict, Any, Any], float],
        device,
    ) -> {}:

        powerset = list(BaseContributionAssessor.generate_power_set(client_index_for_this_round))

        util = {}

        for S in powerset:
            # TODO: this is basically training from scratch with the users in set S and returning the final test accuracy
            util[S] = V_S_D(S=S)

        self.SV = self.shapley_value(util, client_index_for_this_round)

        return self.SV

    def shapley_value(self, utility, client_index_for_this_round):
        N = len(client_index_for_this_round)
        sv_dict = {id: 0 for id in client_index_for_this_round}
        for S in utility.keys():
            if S != ():
                for id in S:
                    marginal_contribution = utility[S] - utility[tuple(i for i in S if i != id)]
                    sv_dict[id] += marginal_contribution / ((comb(N - 1, len(S) - 1)) * N)
        return sv_dict
