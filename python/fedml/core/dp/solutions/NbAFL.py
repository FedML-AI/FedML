from typing import List, Tuple, Dict, Any


class NbAFL_DP():
    def __init__(self, config):
        pass

    def add_local_noise(self, raw_client_grad_list: List[Tuple[float, Dict]],
        extra_auxiliary_info: Any = None):
        # currently use FedMLDifferentialPrivacy to implement this. We will definitely change the class name & code structure later
        pass

    def add_global_noise(
            self,
            global_model,
            extra_auxiliary_info: Any = None,
    ):
        # currently use FedMLDifferentialPrivacy to implement this. We will definitely change the class name & code structure later
        pass

