from .client import client_initializer
from ..core import ClientTrainer


class FedMLModelServingClient:
    """
    Client for Federated Machine Learning Model Serving.

    This class is responsible for initializing and running the client for federated machine learning model serving.

    Args:
        args: An instance of arguments containing configuration settings.
        end_point_name: The name of the model serving endpoint.
        model_name: The name of the machine learning model.
        model_version: The version of the machine learning model.
        inference_request: An optional inference request configuration.
        device: The device (e.g., 'cuda:0') to run the client on.
        dataset: The dataset used for training and testing the model.
        model: The machine learning model to be used.
        model_trainer: An optional client trainer for model training.

    Attributes:
        end_point_name: The name of the model serving endpoint.
        model_name: The name of the machine learning model.
        model_version: The version of the machine learning model.
        inference_request: An optional inference request configuration.

    Methods:
        run(): Start the client for federated machine learning model serving.
    """

    def __init__(self, args, end_point_name, model_name, model_version,
                 inference_request=None, device=None, dataset=None, model=None,
                 model_trainer: ClientTrainer = None):
        """
        Initializes the FedMLModelServingClient.

        Args:
            args: An instance of arguments containing configuration settings.
            end_point_name: The name of the model serving endpoint.
            model_name: The name of the machine learning model.
            model_version: The version of the machine learning model.
            inference_request: An optional inference request configuration.
            device: The device (e.g., 'cuda:0') to run the client on.
            dataset: The dataset used for training and testing the model.
            model: The machine learning model to be used.
            model_trainer: An optional client trainer for model training.
        """
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
            raise Exception("Unsupported federated optimizer")

    def run(self):
        """
        Start the client for federated machine learning model serving.

        This method initializes and runs the client for federated machine learning model serving.

        Returns:
            None
        """
        pass
