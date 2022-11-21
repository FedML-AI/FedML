import logging
import numpy as np
from typing import List, Dict, Callable, Any
import random
import math

from .base_contribution_assessor import BaseContributionAssessor
from .base_contribution_assessor import powersettool
from .base_contribution_assessor import V_S_t


class LeaveOneOut(BaseContributionAssessor):
    def __init__(self):
        super().__init__()

        #trunc paras
        self.round_trunc_threshold=0.01

        self.Contribution_records =[]


    def run(
        self,
        num_client_for_this_round: int,
        idxs: List,
        fraction: Dict,  # this is the weights of the clients in FedAvg
        local_weights_from_clients: List[Dict],
        model_aggregated: Dict,
        model_last_round: Dict,
        acc_on_aggregated_model: float,
        val_dataloader: Any,
        validation_func: Callable[[Dict, Any, Any], float],
        device,
    ) -> List[float]:


        N = num_client_for_this_round
        self.Contribution_records=[]

        powerset = list(powersettool(idxs))

        util={}

        # past iteration's accuracy
        S_0=()
        util[S_0]=validation_func(model_last_round, val_dataloader, device)

        # updated model's accuracy with all participants in the current iteration
        S_all = powerset[-1]
        util[S_all] = acc_on_aggregated_model

        # if not enough improvement in model this iteration, everyone's contributions are 0
        # truncated design

        if abs(util[S_all]-util[S_0]) <= self.round_trunc_threshold:
            contribution_dict = {id:0 for id in idxs} # TO DO: make this a list too?
            return contribution_dict

        marginal_contribution = [0 for i in range(1, N + 1)]

        for j in range(1, N + 1):

            # the idea is to sample a subset of the users of cardinality logN (N being the number of users in this iteration)
            # this sample subset would be randomly selected for each client
            number_user_sampled = round(math.log10(N))  # we do base 10 for now
            if number_user_sampled < 1:
                number_user_sampled = 1

            C_sampled = [j]  # add the user first
            C_sampled.extend(random.sample([x for x in idxs if x != j], number_user_sampled))
            C_sampled = tuple(np.sort(C_sampled, kind='mergesort'))
            print('the considered scenario (C_sampled) is', C_sampled)

            agg_model_C = V_S_t(model_last_round, local_weights_from_clients, fraction, S=C_sampled)
            util[C_sampled] = validation_func(agg_model_C, val_dataloader, device)

            C_removed = [i for i in C_sampled if i != j]
            C_removed = tuple(np.sort(C_removed, kind='mergesort'))
            print('the removed scenario (C_removed) is', C_removed)

            agg_model_C_removed = V_S_t(model_last_round, local_weights_from_clients, fraction, S=C_removed)
            util[C_removed] = validation_func(agg_model_C_removed, val_dataloader, device)

            #print('left out user alone has accuracy', V_S_t(t=t, S=tuple(np.sort([j], kind='mergesort'))))

            # update SV
            marginal_contribution[j-1] = util[C_sampled] - util[C_removed]
            print(marginal_contribution)


        # These are methods to normalize the marginal contributions of the users.

        #marginal_contribution_normalized = [
        #    (float(i) - min(marginal_contribution)) / (max(marginal_contribution) - min(marginal_contribution)) for i in
        #    marginal_contribution]
        #marginal_contribution_normalized = [i / max(np.abs(marginal_contribution)) for i in marginal_contribution]

        print(marginal_contribution)
        self.Contribution_records.append(marginal_contribution)

        shapley_values = (np.cumsum(self.Contribution_records, 0) /
                         np.reshape(np.arange(1, len(self.Contribution_records) + 1), (-1, 1)))[-1:].tolist()[0]

        print(shapley_values)

        return shapley_values # TO DO: return dict?


    '''
            # accuracy of the aggregated model
        acc_on_aggregated_model = validation_func(model_aggregated, val_dataloader, device)
        contributions = np.zeros(num_client_for_this_round, dtype='f')
        for client in range(num_client_for_this_round):
            # assuming same number of samples in each client
            model_aggregated_wo_client = np.sum(i for i in model_list_from_client_update if i != client) / (num_client_for_this_round-1)
            acc_wo_client = validation_func(model_aggregated_wo_client, val_dataloader, device)
            contributions[client] = acc_on_aggregated_model-acc_wo_client
        logging.info("contributions = {}".format(contributions))
        return contributions #[i*0.1 for i in range(num_client_for_this_round)]
    '''

