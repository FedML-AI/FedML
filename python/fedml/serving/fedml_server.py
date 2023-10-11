from fedml.core import ServerAggregator


class FedMLModelServingServer:
    """
    Represents a server for serving federated machine learning models.

    This class initializes and manages the server-side functionality for serving federated models
    in a federated learning system.

    Args:
        args (object): Configuration arguments for the server.
        end_point_name (str): The name of the endpoint for serving the model.
        model_name (str): The name of the federated model.
        model_version (str): The version of the federated model.
        inference_request (object, optional): An inference request object for making predictions.
        device (str, optional): The hardware device to use for inference (e.g., 'cpu' or 'cuda').
        dataset (list, optional): A list containing dataset-related information.
        model (object, optional): The federated machine learning model.
        server_aggregator (ServerAggregator, optional): The server aggregator for model aggregation.

    Methods:
        run(): Starts the server and serves the federated model for inference.

    Note:
        This class is designed for serving federated models in a federated learning system.
    """

    def __init__(self, args, end_point_name, model_name, model_version,
                 inference_request=None, device=None, dataset=None, model=None,
                 server_aggregator: ServerAggregator = None):
        """
        Initializes a Federated Model Serving Server instance.
        """
        self.end_point_name = end_point_name
        self.model_name = model_name
        self.model_version = model_version
        self.inference_request = inference_request

        if args.federated_optimizer == "FedAvg":
            from fedml.cross_silo.server import server_initializer

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
            server_initializer.init_server(
                args,
                device,
                args.comm,
                args.rank,
                args.worker_num,
                model,
                train_data_num,
                train_data_global,
                test_data_global,
                train_data_local_dict,
                test_data_local_dict,
                train_data_local_num_dict,
                server_aggregator,
            )
        else:
            raise Exception("Exception")

    def run(self):
        """
        Starts the server and serves the federated model for inference.
        """
        pass
