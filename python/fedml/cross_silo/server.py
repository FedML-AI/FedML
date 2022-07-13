from .hierarchical.fedml_hierarchical_api import FedML_Hierarchical
from .horizontal.fedml_horizontal_api import FedML_Horizontal
from .horizontal.lsa_fedml_api import FedML_LSA_Horizontal
from ..constants import (
    FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL,
    FEDML_CROSS_SILO_SCENARIO_HORIZONTAL,
)


class Server:
    def __init__(
        self, args, device, dataset, model, model_trainer=None, server_aggregator=None
    ):
        if args.federated_optimizer == "FedAvg":

            if args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
                self.fl_trainer = FedML_Hierarchical(
                    args,
                    0,
                    args.worker_num,
                    args.comm,
                    device,
                    dataset,
                    model,
                    model_trainer=model_trainer,
                    preprocessed_sampling_lists=None,
                )
            elif args.scenario == FEDML_CROSS_SILO_SCENARIO_HORIZONTAL:
                self.fl_trainer = FedML_Horizontal(
                    args,
                    0,
                    args.worker_num,
                    args.comm,
                    device,
                    dataset,
                    model,
                    model_trainer=model_trainer,
                    preprocessed_sampling_lists=None,
                )
        elif args.federated_optimizer == "LSA":
            self.fl_trainer = FedML_LSA_Horizontal(
                args,
                0,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=model_trainer,
                preprocessed_sampling_lists=None,
            )
        else:
            raise Exception("Exception")

    def run(self):
        pass
