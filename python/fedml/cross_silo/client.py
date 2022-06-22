from .horizontal.fedml_horizontal_api import FedML_Horizontal


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
        else:
            raise Exception("Exception")

    def run(self):
        pass
