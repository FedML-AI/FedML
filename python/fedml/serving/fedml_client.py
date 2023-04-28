from .client import client_initializer
from ..core import ClientTrainer


class FedMLModelServingClient:
    def __init__(self, args, end_point_name, model_name, model_version,
                 inference_request=None, device=None, dataset=None, model=None,
                 model_trainer: ClientTrainer = None):
        self.end_point_name = end_point_name
        self.model_name = model_name
        self.model_version = model_version
        self.inference_request = inference_request

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
