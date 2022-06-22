from ....core.alg_frame.params import Params
from .common import fedml_nccl_broadcast
from .common import fedml_nccl_reduce
from .common import fedml_nccl_barrier
from .common import (get_server_rank, get_rank)

from .common import ReduceOp

from .common import (
    move_to_cpu, move_to_gpu,
    clear_optim_buffer, optimizer_to
)

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reduce_params = dict([
            (ReduceOp.SUM, []),
        ])
        self._gather_params = []


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
        new_name = f"client{client_index}_name"
        self.__dict__.update({new_name: param})
        self._gather_params.append(new_name)


    def communicate(self):
        for param_name in self._reduce_params[ReduceOp.SUM]:
            param = getattr(self, param_name)
            if isinstance(param, list):
                for tensor in param:
                    fedml_nccl_reduce(tensor=tensor, dst=get_server_rank())
            else:
                fedml_nccl_reduce(tensor=param, dst=get_server_rank())


        for param_name in self._gather_params:
            pass




class ClientToLocalAggregatorParams(Params):
    def __init__(self, client_index, **kwargs):
        super().__init__(**kwargs)
        self.client_index = client_index
        self._reduce_params = dict([
            (ReduceOp.MEAN, []),
            (ReduceOp.SUM, []),
        ])
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
















