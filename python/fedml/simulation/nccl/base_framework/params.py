import logging

from .common import ReduceOp
from .common import fedml_nccl_broadcast
from .common import fedml_nccl_reduce
from .common import fedml_nccl_send_to_server
from .common import get_server_rank, get_worker_number
from ....core.alg_frame.params import Params


class Params(Params):
    """
    Unified Parameter Object for passing arguments among APIs
            from the algorithm frame (e.g., client_trainer.py and server aggregator.py).

    Usage::
        >> my_params = Params()
        >> # add parameter
        >> my_params.add(name="w", param=model_weights)
        >> # get parameter
        >> my_params.w
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ServerToClientParams(Params):
    """
    Normally, ServerToClient only broadcast parameters, hence all devices will receive same data from server.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._broadcast_params = []
        # self._broadcast_params = {}

    def add_broadcast_param(self, name, param):
        self.__dict__.update({name: param})
        self._broadcast_params.append(name)
        # self._broadcast_params.update({name: param})

    def broadcast(self):
        """
        Perform communication of the added parameters.
        Note that this is a collective operation and all processes (server and devices) must call this function.
        """

        for param_name in self._broadcast_params:
            param = getattr(self, param_name)
            if isinstance(param, list):
                for tensor in param:
                    fedml_nccl_broadcast(tensor=tensor, src=get_server_rank())
            else:
                fedml_nccl_broadcast(tensor=param, src=get_server_rank())


class LocalAggregatorToServerParams(Params):
    # def __init__(self, client_indexes, rank, group, **kwargs):
    def __init__(self, client_indexes, **kwargs):
        """
            client_indexes and group are used to indicate client_indexes that are 
            simulated by currernt LocalAggregator,
            This will be used for gathering data.
        """
        super().__init__(**kwargs)
        self._reduce_params = dict([(ReduceOp.SUM, []),])
        self._gather_params = []
        self.client_indexes = client_indexes
        # self.rank = rank
        # self.group = group

    def add_reduce_param(self, name, param, op=ReduceOp.SUM):
        if name in self.__dict__:
            if isinstance(self.__dict__[name], list):
                for i, tensor in enumerate(param):
                    self.__dict__[name][i] += tensor
            else:
                self.__dict__[name] += param
        else:
            self.__dict__.update({name: param})
            self._reduce_params[op].append(name)

    def add_gather_params(self, client_index, name, param):
        """
            Server needs to add all gather param of all clients,
            Then the collective communication can work.
        """
        # new_name = f"client{client_index}_name"
        # self.__dict__.update({new_name: param})
        # self._gather_params.append(new_name)
        # self.__dict__.update({name: {}})
        if name not in self.__dict__:
            self.__dict__.update({name: {}})
            self._gather_params.append(name)
        self.__dict__[name][client_index] = param

    def communicate(self, rank, groups, client_schedule=None):
        for param_name in self._reduce_params[ReduceOp.SUM]:
            param = getattr(self, param_name)
            if isinstance(param, list):
                for tensor in param:
                    fedml_nccl_reduce(tensor=tensor, dst=get_server_rank())
            else:
                fedml_nccl_reduce(tensor=param, dst=get_server_rank())

        if rank == 0:
            logging.info(f"server:   {client_schedule}, groups: {groups}")
            for param_name in self._gather_params:
                for device in range(get_worker_number()):
                    # logging.info(f"server:  rank:{device}, has client_indexes: {client_schedule[device]}")
                    for client_index in client_schedule[device]:
                        device_rank = device + 1
                        # Here the
                        fedml_nccl_send_to_server(
                            tensor=self.__dict__[param_name][client_index], src=device_rank, group=groups[device_rank]
                        )
        else:
            # logging.info(f"rank:{rank}, groups: {groups}")
            for param_name in self._gather_params:
                # gathered_list = [
                #     torch.empty_like(data) for _ in range(get_world_size())
                # ]
                client_indexes = list(self.__dict__[param_name].keys())
                # logging.info(f"rank:{rank}, has client_indexes: {client_indexes}")
                for client_index, param in self.__dict__[param_name].items():
                    fedml_nccl_send_to_server(param, src=rank, group=groups[rank])


class ClientToLocalAggregatorParams(Params):
    def __init__(self, client_index, **kwargs):
        super().__init__(**kwargs)
        self.client_index = client_index
        self._reduce_params = dict([(ReduceOp.MEAN, []), (ReduceOp.SUM, []),])
        self._gather_params = []

    def add_reduce_param(self, name, param, op=ReduceOp.SUM):
        self.__dict__.update({name: param})
        self._reduce_params[op].append(name)

    def add_gather_params(self, name, param):
        self.__dict__.update({name: param})
        self._gather_params.append(name)

    def get_mean_reduce_param_names(self):
        return self._reduce_params[ReduceOp.MEAN]

    def get_sum_reduce_param_names(self):
        return self._reduce_params[ReduceOp.SUM]

    def get_gather_param_names(self):
        return self._gather_params


def local_gather(local_gather_params):
    pass
