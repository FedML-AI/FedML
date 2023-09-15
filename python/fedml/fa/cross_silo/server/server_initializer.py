from .fedml_aggregator import FAAggregator
from .fedml_server_manager import FedMLServerManager
from ...aggregator.global_analyzer_creator import create_global_analyzer


def init_server(
    args,
    comm,
    rank,
    worker_num,
    train_data_num,
    train_data_local_dict,
    train_data_local_num_dict,
    server_aggregator,
):
    """
    Initialize the Federated Learning server for Cross-Silo Federated Learning.

    Args:
        args (object): An object containing server configuration parameters.
        comm: The communication object.
        rank (int): The rank of the server.
        worker_num (int): The total number of workers.
        train_data_num (int): The total number of training data samples.
        train_data_local_dict (dict): A dictionary of client-specific training data.
        train_data_local_num_dict (dict): A dictionary of client-specific training data sizes.
        server_aggregator: An instance of the server aggregator (optional).

    Returns:
        None
    """
    if server_aggregator is None:
        server_aggregator = create_global_analyzer(args, train_data_num=train_data_num)
    server_aggregator.set_id(0)

    # aggregator
    aggregator = FAAggregator(
        train_data_num,
        train_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        args,
        server_aggregator,
    )

    # start the distributed training
    backend = args.backend
    server_manager = FedMLServerManager(args, aggregator, comm, rank, worker_num, backend)
    server_manager.run()
