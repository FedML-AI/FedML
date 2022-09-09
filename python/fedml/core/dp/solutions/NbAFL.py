from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism
from fedml.core.dp.solutions.base_dp_solution import BaseDPSolution


class NbAFL(BaseDPSolution):
    def __init__(self, args):
        super().__init__(args)
        self.set_cdp(
            DPMechanism(
                args.cdp_mechanism_type,
                args.cdp_epsilon,
                args.cdp_delta,
                args.cdp_sensitivity,
            )
        )

        self.set_ldp(
            DPMechanism(
                args.ldp_mechanism_type,
                args.ldp_epsilon,
                args.ldp_delta,
                args.ldp_sensitivity,
            )
        )

    def add_local_noise(self, local_grad: dict):
        return super().add_local_noise(local_grad=local_grad)

    def add_global_noise(self, global_model: dict):
        return super().add_global_noise(global_model=global_model)
