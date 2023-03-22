from .LocalAggregator import BaseLocalAggregator
from .Server import BaseServer


def FedML_Base_NCCL(args, process_id, worker_number, comm, device, dataset, model, model_trainer=None):

    if process_id == 0:
        return BaseServer(args, process_id, worker_number, comm, device, dataset, model, model_trainer)
    else:
        return BaseLocalAggregator(args, process_id, worker_number, comm, device, dataset, model, model_trainer)
