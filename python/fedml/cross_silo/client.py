from .horizontal.fedml_horizontal_api import FedML_Horizontal
from .horizontal.lsa_fedml_api import FedML_LSA_Horizontal

class Client:
    def __init__(self, args, device, dataset, model, model_trainer=None):
        if args.federated_optimizer == "FedAvg":
            self.fl_trainer = FedML_Horizontal(
                args,
                args.rank,  # Note: client rank stars from 1
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
                args.rank,
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
