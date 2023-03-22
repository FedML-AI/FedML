from ..base_framework.LocalAggregator import BaseLocalAggregator


class FedAvgLocalAggregator(BaseLocalAggregator):
    """
    Used to manage and aggregate results from local trainers (clients).
    It needs to know all datasets.
    device: indicates the device of this local aggregator.
    """

    # def __init__(self, args, trainer, device, dataset, comm=None, rank=0, size=0, backend="NCCL"):
    #     super().__init__(args, args, trainer, device, dataset, comm, rank, size, backend)
    def __init__(self, args, rank, worker_number, comm, device, dataset, model, trainer):
        super().__init__(args, rank, worker_number, comm, device, dataset, model, trainer)
