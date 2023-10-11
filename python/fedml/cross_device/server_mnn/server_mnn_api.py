import logging

from .fedml_aggregator import FedMLAggregator
from .fedml_server_manager import FedMLServerManager
from ...ml.aggregator.aggregator_creator import create_server_aggregator


def fedavg_cross_device(args, process_id, worker_number, comm, device, test_dataloader, model, server_aggregator=None):
    """
    Federated Averaging across Multiple Devices (Cross-Device Aggregation).

    This function performs federated averaging across multiple devices using cross-device aggregation.

    Args:
        args: Arguments for the federated learning process.
        process_id (int): The process ID of the current worker.
        worker_number (int): The total number of workers.
        comm: Communication backend for distributed training.
        device: The device (e.g., CPU or GPU) to perform computations.
        test_dataloader: DataLoader for the test dataset.
        model: The federated learning model.
        server_aggregator: Server aggregator for aggregating model updates (default: None).

    Returns:
        None
    """
    logging.info("test_data_global.iter_number = {}".format(
        test_dataloader.iter_number))

    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number,
                    model, test_dataloader, server_aggregator)


def init_server(args, device, comm, rank, size, model, test_dataloader, aggregator):
    """
    Initialize the Federated Learning Server.

    This function initializes the federated learning server for aggregation.

    Args:
        args: Arguments for the federated learning process.
        device: The device (e.g., CPU or GPU) to perform computations.
        comm: Communication backend for distributed training.
        rank (int): The rank of the current worker.
        size (int): The total number of workers.
        model: The federated learning model.
        test_dataloader: DataLoader for the test dataset.
        aggregator: Server aggregator for aggregating model updates.

    Returns:
        None
    """
    if aggregator is None:
        aggregator = create_server_aggregator(model, args)
    aggregator.set_id(-1)

    td_id = id(test_dataloader)
    logging.info("test_dataloader = {}".format(td_id))
    logging.info("test_data_global.iter_number = {}".format(
        test_dataloader.iter_number))

    worker_num = size
    aggregator = FedMLAggregator(
        test_dataloader, worker_num, device, args, aggregator)

    # Start the distributed training
    backend = args.backend
    server_manager = FedMLServerManager(
        args, aggregator, comm, rank, size, backend)
    if not args.using_mlops:
        server_manager.start_train()
    server_manager.run()
