from ..base_framework.Server import BaseServer


class FedAvgServer(BaseServer):
    # def __init__(self, args, trainer, device, dataset, comm=None, rank=0, size=0, backend="NCCL"):
    #     super().__init__(args, args, trainer, device, dataset, comm, rank, size, backend)
    def __init__(self, args, rank, worker_number, comm, device, dataset, model, trainer):
        super().__init__(args, rank, worker_number, comm, device, dataset, model, trainer)
