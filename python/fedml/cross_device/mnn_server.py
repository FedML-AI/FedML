import logging

from .server_mnn.server_mnn_api import fedavg_cross_device


class ServerMNN:
    """
    A class representing the server in federated learning using MNN (Mobile Neural Networks).

    This class is responsible for coordinating and aggregating model updates from client devices.

    Args:
        args: The command-line arguments.
        device: The device for computations.
        test_dataloader: The DataLoader for testing data.
        model: The federated learning model.
        server_aggregator: The server aggregator (optional).

    Attributes:
        None

    Methods:
        run: Run the server for federated learning.
    """

    def __init__(self, args, device, test_dataloader, model, server_aggregator=None):
        """
        Initialize a ServerMNN instance.

        Args:
            args: The command-line arguments.
            device: The device for computations.
            test_dataloader: The DataLoader for testing data.
            model: The federated learning model.
            server_aggregator: The server aggregator (optional).
        """
        if args.federated_optimizer == "FedAvg":
            logging.info("test_data_global.iter_number = {}".format(
                test_dataloader.iter_number))

            fedavg_cross_device(
                args, 0, args.worker_num, None, device, test_dataloader, model, server_aggregator=server_aggregator
            )
        else:
            raise Exception("Unsupported federated optimizer")

    def run(self):
        """
        Run the server for federated learning.

        This method coordinates and aggregates model updates from client devices.
        """
        pass
