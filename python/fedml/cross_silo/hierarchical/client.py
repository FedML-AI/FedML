from .fedml_hierarchical_api import FedML_Hierarchical
import os


class Client:
    def __init__(
        self, args, device, dataset, model, model_trainer=None, process_rank=0
    ):
        if args.federated_optimizer == "FedAvg":
            self.fl_trainer = FedML_Hierarchical(
                args,
                args.rank,  # Note: client rank stars from 1
                args.worker_num,
                None,
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
