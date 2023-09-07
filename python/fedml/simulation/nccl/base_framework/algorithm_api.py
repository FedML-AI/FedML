from .LocalAggregator import BaseLocalAggregator
from .Server import BaseServer


def FedML_Base_NCCL(args, process_id, worker_number, comm, device, dataset, model, model_trainer=None):
    """
    Create an instance of either the BaseServer or BaseLocalAggregator based on the process ID.

    Args:
        args: The arguments for configuring the FedML engine.
        process_id (int): The ID of the current process.
        worker_number (int): The total number of workers in the simulation.
        comm: The communication backend (e.g., MPI communicator).
        device: The device on which the model should be placed.
        dataset: The dataset used for training.
        model: The model to be trained.
        model_trainer: An optional trainer for the model.

    Returns:
        BaseServer or BaseLocalAggregator: An instance of either the server or local aggregator based on the process ID.
    """

    if process_id == 0:
        return BaseServer(args, process_id, worker_number, comm, device, dataset, model, model_trainer)
    else:
        return BaseLocalAggregator(args, process_id, worker_number, comm, device, dataset, model, model_trainer)
