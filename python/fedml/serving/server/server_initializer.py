from .fedml_aggregator import FedMLAggregator
from .fedml_server_manager import FedMLServerManager
from ...ml.aggregator.aggregator_creator import create_server_aggregator


def init_server(
    args,
    device,
    comm,
    rank,
    worker_num,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_dict,
    test_data_local_dict,
    train_data_local_num_dict,
    server_aggregator=None,
):
    """
    Initialize and start the server for federated machine learning.

    Args:
        args: Configuration arguments for the server.
        device: The device (e.g., GPU) to be used for computation.
        comm: Communication module for distributed computing.
        rank: The rank of the server in the communication group.
        worker_num: The number of worker nodes in the federated setup.
        model: The machine learning model to be used.
        train_data_num: The number of training data samples.
        train_data_global: The global training dataset.
        test_data_global: The global test dataset.
        train_data_local_dict: Dictionary of local training datasets for workers.
        test_data_local_dict: Dictionary of local test datasets for workers.
        train_data_local_num_dict: Dictionary of the number of local training samples for workers.
        server_aggregator: The aggregator responsible for aggregating model updates (default: None).
    """
    if server_aggregator is None:
        server_aggregator = create_server_aggregator(model, args)
    server_aggregator.set_id(0)

    # Create the aggregator
    aggregator = FedMLAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        server_aggregator,
    )

    # Start the distributed training
    backend = args.backend
    server_manager = FedMLServerManager(args, aggregator, comm, rank, worker_num, backend)
    server_manager.run()
