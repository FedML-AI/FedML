from typing import List, Dict, Callable, Any

import numpy as np

from .base_contribution_assessor import BaseContributionAssessor


class GTGShapleyValue(BaseContributionAssessor):
    def __init__(self, args):
        super().__init__()

        self.args = args

        # trunc paras
        self.eps = 0.001
        self.round_trunc_threshold = 0.01

        self.Contribution_records = []

        # converge paras
        self.CONVERGE_MIN_K = 3 * 10
        self.last_k = 10
        self.CONVERGE_CRITERIA = 0.05

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

        # truncation for negative accuracy improvement:
        # if not enough improvement in model this iteration, everyone's contributions are 0
        if abs(util[s_all] - util[s_0]) <= self.round_trunc_threshold:
            contribution_dict = [0 for id in client_index_for_this_round]
            return contribution_dict

        k = 0
        while self._is_not_converged(k):
            for pi in client_index_for_this_round:
                k += 1
                v = [0 for i in range(N + 1)]
                v[0] = util[s_0]
                marginal_contribution_k = [0 for i in range(N)]

                idxs_k = np.concatenate(
                    (np.array([pi]), np.random.permutation([p for p in client_index_for_this_round if p != pi]))
                )
                print(idxs_k)
                for j in range(1, N + 1):
                    # key = C subset
                    C = idxs_k[:j]
                    C = tuple(np.sort(C, kind="mergesort"))

                    # truncation
                    if abs(util[s_all] - v[j - 1]) >= self.eps:
                        if util.get(C) != None:
                            v[j] = util[C]
                        else:
                            agg_model_C = BaseContributionAssessor.get_aggregated_model_with_client_subset(
                                self.args, aggregation_func, local_weights_from_clients, C
                            )
                            v[j] = validation_func(agg_model_C, val_dataloader, device)
                    else:
                        v[j] = v[j - 1]

                    # record calculated V(C)
                    util[C] = v[j]
                    # update SV
                    marginal_contribution_k[idxs_k[j - 1] - 1] = v[j] - v[j - 1]

                self.Contribution_records.append(marginal_contribution_k)

        # shapley value calculation
        shapley_values = (
            np.cumsum(self.Contribution_records, 0)
            / np.reshape(np.arange(1, len(self.Contribution_records) + 1), (-1, 1))
        )[-1:].tolist()[0]

        for index, client_ind in enumerate(client_index_for_this_round):
            self.shapley_values_by_round[self.args.round_idx][client_ind] = shapley_values[index]

    def _is_not_converged(self, k):

        if k <= self.CONVERGE_MIN_K:
            return True
        all_vals = (
            np.cumsum(self.Contribution_records, 0)
            / np.reshape(np.arange(1, len(self.Contribution_records) + 1), (-1, 1))
        )[-self.last_k :]
        # errors = np.mean(np.abs(all_vals[-last_K:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
        errors = np.mean(np.abs(all_vals[-self.last_k :] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + 1e-12), -1)
        if np.max(errors) > self.CONVERGE_CRITERIA:
            return True
        return False

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
