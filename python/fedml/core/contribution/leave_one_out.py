import math
import random
from typing import List, Dict, Callable, Any

import numpy as np

from .base_contribution_assessor import BaseContributionAssessor


class LeaveOneOut(BaseContributionAssessor):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # trunc paras
        self.round_trunc_threshold = 0.01

        self.Contribution_records = []

        # key is the round_index;
        # value is the dictionary contribution (key - client_index; value - relative contribution
        self.shapley_values_by_round = dict()
        self.SV = {} # dict: {id:SV,...}
        self.SV_summed = {} # dict: {id: SV_summed_over_all_iter, ...}

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

        N = num_client_for_this_round
        self.Contribution_records = []

        powerset = list(BaseContributionAssessor.generate_power_set(client_index_for_this_round))

        util = {}

        # past iteration's accuracy
        s_0 = ()
        util[s_0] = acc_on_last_round

        # updated model's accuracy with all participants in the current iteration
        s_all = powerset[-1]
        util[s_all] = acc_on_aggregated_model

        # if not enough improvement in model this iteration, everyone's contributions are 0
        # truncated design

        if abs(util[s_all] - util[s_0]) <= self.round_trunc_threshold:
            contribution_dict = [0 for id in client_index_for_this_round]
            return contribution_dict

        marginal_contribution = [0 for i in range(1, N + 1)]

        for j in range(N):
            '''
            the idea is to sample a subset of the users of cardinality logN 
            (N being the number of users in this iteration)
            this sample subset would be randomly selected for each client
            '''
            number_user_sampled = round(math.log10(N))
            if number_user_sampled < 1:
                number_user_sampled = 1

            C_sampled = [client_index_for_this_round[j]]  # add the user first
            C_sampled.extend(random.sample([x for x in client_index_for_this_round if x != j], number_user_sampled))
            C_sampled = tuple(np.sort(C_sampled, kind="mergesort"))
            print("the considered scenario (C_sampled) is", C_sampled)

            agg_model_C = BaseContributionAssessor.get_aggregated_model_with_client_subset(
                                self.args, aggregation_func, local_weights_from_clients, C_sampled
                            )
            util[C_sampled] = validation_func(agg_model_C, val_dataloader, device)

            C_removed = [i for i in C_sampled if i != j]
            C_removed = tuple(np.sort(C_removed, kind="mergesort"))
            print("the removed scenario (C_removed) is", C_removed)

            agg_model_C_removed = BaseContributionAssessor.get_aggregated_model_with_client_subset(
                                self.args, aggregation_func, local_weights_from_clients, C_removed
                            )
            util[C_removed] = validation_func(agg_model_C_removed, val_dataloader, device)

            # update SV
            marginal_contribution[j] = util[C_sampled] - util[C_removed]

        self.Contribution_records.append(marginal_contribution)

        shapley_values = (
            np.cumsum(self.Contribution_records, 0)
            / np.reshape(np.arange(1, len(self.Contribution_records) + 1), (-1, 1))
        )[-1:].tolist()[0]

        i = 0
        for client_ind in client_index_for_this_round:
            self.shapley_values_by_round[self.args.round_idx][client_ind] = shapley_values[i]
            i = i + 1

    def get_final_contribution_assignment(self):
        """
        return: contribution_assignment
            (key is client_index; value is the client's relative contribution);
            the sum of values in the dictionary is equal to 1
        """
        for t, shapley_t in self.shapley_values_by_round.items():
            for id in shapley_t:
                if self.SV.get(id):
                    self.SV[id].append(shapley_t[id])
                else:
                    self.SV[id] = [shapley_t[id]]

        contribution_assignment = dict()
        keys = range(1, self.args.client_num_in_total+1)
        for id in keys:
            self.SV_summed[id] = np.sum(self.SV[id])
        total_sv = sum(self.SV_summed.values())

        for id in keys:
            contribution_assignment[id] = self.SV_summed[id]/total_sv
        return contribution_assignment