from fedml.ml.trainer.trainer_creator import create_model_trainer
from .LocalAggregator import FedAvgLocalAggregator
from .Server import FedAvgServer


def FedML_FedAvg_NCCL(args, process_id, worker_number, comm, device, dataset, model, model_trainer=None):
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
