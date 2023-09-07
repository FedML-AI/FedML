from fedml.ml.trainer.trainer_creator import create_model_trainer
from .LocalAggregator import FedAvgLocalAggregator
from .Server import FedAvgServer


def FedML_FedAvg_NCCL(args, process_id, worker_number, comm, device, dataset, model, model_trainer=None):
    """
    Create a FedAvgServer or FedAvgLocalAggregator object based on the process ID.

    This function is a factory function for creating either a FedAvgServer or a FedAvgLocalAggregator object
    based on the value of the process ID. If the process ID is 0, it creates a FedAvgServer object; otherwise,
    it creates a FedAvgLocalAggregator object.

    Args:
        args (object): Arguments for the federated learning setup.
        process_id (int): The process ID.
        worker_number (int): The total number of worker processes.
        comm (object): The communication backend.
        device (object): The device on which the model is trained.
        dataset (tuple): A tuple containing dataset-related information.
        model (object): The machine learning model.
        model_trainer (object, optional): The model trainer. If not provided, it will be created.

    Returns:
        object: A FedAvgServer or FedAvgLocalAggregator object.
    """
    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)
    if process_id == 0:
        return FedAvgServer(args, process_id, worker_number, comm, device, dataset, model, model_trainer)
    else:
        return FedAvgLocalAggregator(args, process_id, worker_number, comm, device, dataset, model, model_trainer)


# def FedML_FedAvg_NCCL(args, process_id, worker_number, comm):
#     if process_id == 0:
#         return init_server(args, comm, process_id, worker_number)
#     else:
#         return init_local_aggregator(args, comm, process_id, worker_number)


# def init_server(args, comm, process_id, size):
#     # aggregator
#     client_num = size - 1
#     server = FedAvgServer(client_num, args)
#     return server


# def init_local_aggregator(args, comm, process_id, size):
#     # trainer
#     client_ID = process_id - 1
#     local_aggregator = FedAvgLocalAggregator()
#     return local_aggregator
