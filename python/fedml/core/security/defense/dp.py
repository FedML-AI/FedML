from typing import Callable, List, Tuple, Dict, Any
from ...differential_privacy.fed_privacy_mechanism import DP


class DifferentialPrivacy:
    def __init__(self, config):
        if config.dp_type not in ["cdp", "ldp"]:
            raise ValueError("DP type %s does not exist" % config.dp_type)
        self.dp = DP(
            config.epsilon, config.delta, config.sensitivity, config.mechanism_type
        )
        self.dp_type = config.dp_type

    def run(
        self,
        raw_client_grad_list: List[Tuple[float, Dict]],
        base_aggregation_func: Callable = None,
        extra_auxiliary_info: Any = None,
    ) -> Dict:
        if self.dp_type == "ldp":
            client_grad_list = []
            for (sample_num, client_grad) in raw_client_grad_list:
                client_grad_list.append(
                    (sample_num, self.dp.compute_randomized_gradient(client_grad))
                )
            return base_aggregation_func(client_grad_list)
        else:
            result = base_aggregation_func(raw_client_grad_list)
            return self.dp.compute_randomized_gradient(result)
