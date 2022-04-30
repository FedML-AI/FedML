from .horizontal.fedml_horizontal_api import FedML_Horizontal


class Server:
    def __init__(self, args, device, dataset, model, model_trainer=None):
        if args.federated_optimizer == "FedAvg":
            self.fl_trainer = FedML_Horizontal(
                args,
                0,
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
