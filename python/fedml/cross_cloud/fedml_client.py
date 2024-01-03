from .client import client_initializer
from ..core import ClientTrainer


class FedMLCrossCloudClient:
    def __init__(self, args, device, dataset, model, model_trainer: ClientTrainer = None):
        if args.federated_optimizer == "FedAvg":
            [
                train_data_num,
                test_data_num,
                train_data_global,
                test_data_global,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                class_num,
            ] = dataset
            client_initializer.init_client(
                args,
                device,
                args.comm,
                args.rank,
                args.worker_num,
                model,
                train_data_num,
                train_data_local_num_dict,
                train_data_local_dict,
                test_data_local_dict,
                model_trainer,
            )

        else:
            raise Exception("Exception")

    def run(self):
        pass
