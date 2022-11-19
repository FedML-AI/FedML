import logging
import numpy as np
from typing import List, Dict, Callable, Any
import random
import math

from .base_contribution_assessor import BaseContributionAssessor
from .base_contribution_assessor import powersettool
from .base_contribution_assessor import V_S_t


class LeaveOneOut(BaseContributionAssessor):
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


        N = num_client_for_this_round  # N = len(idxs)
        # this is a threshold used to
        round_trunc_threshold = 0.01

        eps = 0.001
        Contribution_records=[]

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

        if abs(util[S_all]-util[S_0]) <= round_trunc_threshold:
            contribution_dict = {id:0 for id in idxs} # TO DO: make this a list too?
            return contribution_dict

        k=0
        while self.isnotconverge(self,k):
            for pi in idxs:
                k+=1
                v=[0 for i in range(N+1)]
                v[0]=util[S_0]
                marginal_contribution_k=[0 for i in range(N)]


                idxs_k=np.concatenate((np.array([pi]),np.random.permutation([p for p in idxs if p !=pi])))
                print(idxs_k)
                for j in range(1,N+1):
                    # key = C subset
                    C=idxs_k[:j]
                    C=tuple(np.sort(C,kind='mergesort'))

                    #truncation
                    if abs(util[S_all] - v[j-1])>=eps:
                        if util.get(C)!= None:
                            v[j]=util[C]
                        else:
                            agg_model_C = V_S_t(model_last_round, local_weights_from_clients, fraction, S=C)
                            v[j]= validation_func(agg_model_C, val_dataloader, device)
                    else:
                        v[j]=v[j-1]

                    # record calculated V(C)
                    util[C] = v[j]
                    # update SV
                    marginal_contribution_k[idxs_k[j-1]-1] = v[j] - v[j-1]

                self.Contribution_records.append(marginal_contribution_k)

        # shapley value calculation
        shapley_values = (np.cumsum(self.Contribution_records, 0)/
                         np.reshape(np.arange(1, len(self.Contribution_records)+1), (-1,1)))[-1:].tolist()[0]


        return shapley_values


    def isnotconverge(self,k):

        if k <= self.CONVERGE_MIN_K:
            return True
        all_vals=(np.cumsum(self.Contribution_records, 0)/
                  np.reshape(np.arange(1, len(self.Contribution_records)+1), (-1,1)))[-self.last_k:]
        #errors = np.mean(np.abs(all_vals[-last_K:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
        errors = np.mean(np.abs(all_vals[-self.last_k:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
        if np.max(errors) > self.CONVERGE_CRITERIA:
            return True
        return False