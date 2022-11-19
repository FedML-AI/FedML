from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any
from itertools import chain, combinations
import copy

class BaseContributionAssessor(ABC):

    @abstractmethod
    def run(
        self,
        num_client_for_this_round: int,
        idxs: List, # this is the indices of the participating users in that iteration
        fraction: Dict, # this is the weights of the clients in FedAvg
        local_weights_from_clients: List[Dict],    # TO DO dict: [id]
        model_aggregated: Dict,
        model_last_round: Dict,
        acc_on_aggregated_model: float,
        val_dataloader: Any,
        validation_func: Callable[[Dict, Any, Any], float],
        device,
    ) -> List[float]:
        pass

    def powersettool(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    # Constructs an aggregate model from local updates of the users.
    def V_S_t(self,model_last_round, local_weights_from_clients, fraction, **kwargs):


        S=kwargs['S']
        model_S=copy.deepcopy(model_last_round)
        model_S.train()

        #if S==():
        #    test_acc=self.null_M_acc[t]
        #    return test_acc

        local_weights = {id:local_weights_from_clients[id] for id in S}
        # the second argument below in self.fedavg is the weight of aggregation
        global_weights = self.fedavg(local_weights, {i:fraction[i] for i in S})
        model_S.load_state_dict(global_weights)

        return model_S


# Here fraction is a Dict.
# self.fraction = {i: len(self.clients[i].idxs)/self.total_data for i in range(1, self.N+1)}
    def fedavg(self, w: dict, fraction: dict):
        counter=0
        for id in w.keys():
            counter += 1
            if counter == 1:
                w_avg = copy.deepcopy(w[id])
                for key in w_avg.keys():
                    w_avg[key] *= (fraction[id]/sum(fraction.values()))
            else:
                for key in w_avg.keys():
                    w_avg[key] += w[id][key]*(fraction[id]/sum(fraction.values()))
        return w_avg



