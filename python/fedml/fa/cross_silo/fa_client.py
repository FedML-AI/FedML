from .client import client_initializer
from ..base_frame.client_analyzer import FAClientAnalyzer


class FACrossSiloClient:
    """
    Federated Learning Client for Cross-Silo Federated Learning.

    Args:
        args (object): An object containing client configuration parameters.
        dataset (tuple): A tuple containing dataset information, including size and partitions.
        client_analyzer (FAClientAnalyzer): An instance of the client analyzer (optional).

    Attributes:
        args (object): An object containing client configuration parameters.
        dataset (tuple): A tuple containing dataset information, including size and partitions.
        client_analyzer (FAClientAnalyzer): An instance of the client analyzer.

    Methods:
        run():
            Start the Cross-Silo Federated Learning client.

    """
    def __init__(self, args, dataset, client_analyzer: FAClientAnalyzer = None):
        """
        Initialize the Cross-Silo Federated Learning client.

        Args:
            args (object): An object containing client configuration parameters.
            dataset (tuple): A tuple containing dataset information, including size and partitions.
            client_analyzer (FAClientAnalyzer): An instance of the client analyzer (optional).

        Note:
            This constructor sets up the client and initializes it with the provided dataset and configuration.

        Returns:
            None
        """
        [
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
        ] = dataset
        client_initializer.init_client(
            args,
            args.comm,
            args.rank,
            args.worker_num,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            client_analyzer,
        )

    def run(self):
        """
        Start the Cross-Silo Federated Learning client.

        Returns:
            None
        """
        pass
