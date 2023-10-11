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
    server_aggregator,
):
    """
    Initialize the server for federated learning.

    This function sets up the server for federated learning, including creating an aggregator,
    starting distributed training, and running the server manager.

    Args:
        args (argparse.Namespace): Command-line arguments and configurations.
        device (torch.device): The device on which the server runs.
        comm (Communicator): The communication backend.
        rank (int): The rank of the server in the distributed environment.
        worker_num (int): The number of worker nodes participating in federated learning.
        model (torch.nn.Module): The model used for federated learning.
        train_data_num (int): The number of training data points globally.
        train_data_global (Dataset): The global training dataset.
        test_data_global (Dataset): The global test dataset.
        train_data_local_dict (dict): A dictionary of local training datasets for each client.
        test_data_local_dict (dict): A dictionary of local test datasets for each client.
        train_data_local_num_dict (dict): A dictionary of the number of local training data points for each client.
        server_aggregator (ServerAggregator, optional): The server aggregator. If not provided, it will be created.

    Returns:
        None
    """
    if server_aggregator is None:
        server_aggregator = create_server_aggregator(model, args)
    server_aggregator.set_id(0)

    # aggregator
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

    # start the distributed training
    backend = args.backend
    server_manager = FedMLServerManager(
        args, aggregator, comm, rank, worker_num, backend)
    server_manager.run()
