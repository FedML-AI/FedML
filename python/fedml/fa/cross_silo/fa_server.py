from fedml.fa.base_frame.server_aggregator import FAServerAggregator
from fedml.fa.cross_silo.server.server_initializer import init_server


class FACrossSiloServer:
    """
    Federated Learning Server for Cross-Silo Federated Learning.

    Args:
        args (object): An object containing server configuration parameters.
        dataset (tuple): A tuple containing dataset information, including size and partitions.
        server_aggregator (FAServerAggregator): An instance of the server aggregator (optional).

    Attributes:
        args (object): An object containing server configuration parameters.
        dataset (tuple): A tuple containing dataset information, including size and partitions.
        server_aggregator (FAServerAggregator): An instance of the server aggregator.

    Methods:
        run():
            Start the Cross-Silo Federated Learning server.

    """
    def __init__(self, args, dataset, server_aggregator: FAServerAggregator = None):
        """
        Initialize the Cross-Silo Federated Learning server.

        Args:
            args (object): An object containing server configuration parameters.
            dataset (tuple): A tuple containing dataset information, including size and partitions.
            server_aggregator (FAServerAggregator): An instance of the server aggregator (optional).

        Note:
            This constructor sets up the server and initializes it with the provided dataset and configuration.

        Returns:
            None
        """
        [
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
        ] = dataset
        init_server(
            args,
            args.comm,
            args.rank,
            args.worker_num,
            train_data_num,
            train_data_local_dict,
            train_data_local_num_dict,
            server_aggregator,
        )

    def run(self):
        """
        Start the Cross-Silo Federated Learning server.

        Returns:
            None
        """
        pass
