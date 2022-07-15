from .client import client_initializer
from .lightsecagg.lsa_fedml_api import FedML_LSA_Horizontal


class Client:
    def __init__(
        self, args, device, dataset, model, model_trainer=None, server_aggregator=None
    ):
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
            self.fl_trainer = client_initializer.init_client(
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

