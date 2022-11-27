from typing import List, Dict, Callable, Any

import numpy as np
from scipy.special import comb

from .base_contribution_assessor import BaseContributionAssessor


class MRShapleyValue(BaseContributionAssessor):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # trunc paras
        self.eps = 0.001
        self.round_trunc_threshold = 0.01

        self.Contribution_records = []

        # key is the round_index;
        # value is the dictionary contribution (key - client_index; value - relative contribution
        self.shapley_values_by_round = dict()
        self.SV = {}  # dict: {id:SV,...}
        self.SV_summed = {}  # dict: {id: SV_summed_over_all_iter, ...}

    def run(
            self,
            num_client_for_this_round: int,  # e.g, select 4 from 8
            client_index_for_this_round: List,  # e.g., 4 selected clients from 8 clients [1, 3, 4, 7]
            aggregation_func: Callable,
            local_weights_from_clients: List[Dict],
            acc_on_last_round: float,
            acc_on_aggregated_model: float,
            val_dataloader: Any,
            validation_func: Callable[[Dict, Any, Any], float],
            device,
    ):
        # set the model first and then evaluate
        # (metric1, metric2, metric3, metric4) = validation_func(self.test_global, device, self.args)

        util = {}

        powerset = list(BaseContributionAssessor.generate_power_set(client_index_for_this_round))

        for S in powerset:
            agg_model_with_subset_S = BaseContributionAssessor.get_aggregated_model_with_client_subset(
                self.args, aggregation_func, local_weights_from_clients, S
            )
            util[S] = validation_func(agg_model_with_subset_S, val_dataloader, device)

        self.shapley_values_by_round[self.args.round_idx] = self.shapley_value(util, client_index_for_this_round)

    def shapley_value(self, utility, idxs):
        N = len(idxs)
        sv_dict = {id: 0 for id in idxs}
        for S in utility.keys():
            if S != ():
                for id in S:
                    marginal_contribution = utility[S] - utility[tuple(i for i in S if i != id)]
                    sv_dict[id] += marginal_contribution / ((comb(N - 1, len(S) - 1)) * N)
        return sv_dict

    def get_final_contribution_assignment(self):
        """
        return: contribution_assignment
            (key is client_index; value is the client's relative contribution);
            the sum of values in the dictionary is equal to 1
        """
        contribution_assignment = dict()

        for t, shapley_t in self.shapley_values_by_round.items():
            for id in shapley_t:
                if self.SV.get(id):
                    self.SV[id].append(shapley_t[id])
                else:
                    self.SV[id] = [shapley_t[id]]

        keys = range(1, self.args.client_num_in_total + 1)

        for id in keys:
            self.SV_summed[id] = np.sum(self.SV[id])
        total_sv = sum(self.SV_summed.values())

        for id in keys:
            contribution_assignment[id] = self.SV_summed[id] / total_sv

        return contribution_assignment
