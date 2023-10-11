import logging

from .common import ReduceOp
from .common import fedml_nccl_broadcast
from .common import fedml_nccl_reduce
from .common import fedml_nccl_send_to_server
from .common import get_server_rank, get_worker_number
from ....core.alg_frame.params import Params


class Params(Params):
    """
    Unified Parameter Object for passing arguments among APIs.

    This class is used for passing arguments among different parts of the algorithm framework.
    You can add parameters and retrieve them using attribute access.

    Example:
        >> my_params = Params()
        >> # Add a parameter
        >> my_params.add(name="w", param=model_weights)
        >> # Get a parameter
        >> weight = my_params.w

    Attributes:
        _params (dict): A dictionary to store parameter names and values.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ServerToClientParams(Params):
    """
    Parameters sent from server to clients for broadcasting.

    This class represents parameters that are broadcasted from the server to all clients.
    It allows adding broadcast parameters and performing the broadcasting operation.

    Example:
        >> server_params = ServerToClientParams()
        >> # Add a broadcast parameter
        >> server_params.add_broadcast_param(name="w", param=model_weights)
        >> # Broadcast the added parameters to all clients
        >> server_params.broadcast()

    Attributes:
        _broadcast_params (list): A list of parameter names to be broadcasted.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._broadcast_params = []
        # self._broadcast_params = {}

    def add_broadcast_param(self, name, param):
        """
        Add a parameter to be broadcasted to all clients.

        Args:
            name (str): The name of the parameter.
            param (torch.Tensor or list of torch.Tensor): The parameter to be broadcasted.

        Returns:
            None
        """
        self.__dict__.update({name: param})
        self._broadcast_params.append(name)
        # self._broadcast_params.update({name: param})

    def broadcast(self): 
        """
        Perform broadcasting of the added parameters to all clients.

        Note:
            This is a collective operation, and all processes (server and devices) must call this function.
        """

        for param_name in self._broadcast_params:
            param = getattr(self, param_name)
            if isinstance(param, list):
                for tensor in param:
                    fedml_nccl_broadcast(tensor=tensor, src=get_server_rank())
            else:
                fedml_nccl_broadcast(tensor=param, src=get_server_rank())


class LocalAggregatorToServerParams(Params):
    """
    Parameters sent from local aggregator to the server for aggregation.

    This class represents parameters that are sent from local aggregators to the server
    for aggregation and communication between clients and the server.

    Attributes:
        _reduce_params (dict): A dictionary containing lists of parameters to be reduced using different operations.
        _gather_params (list): A list of parameter names to be gathered from clients.
        client_indexes (list): List of client indexes for which this local aggregator has data.
    """"
    # def __init__(self, client_indexes, rank, group, **kwargs):
    def __init__(self, client_indexes, **kwargs):
        """
        Initialize the LocalAggregatorToServerParams object.

        Args:
            client_indexes (list): List of client indexes that are simulated by this LocalAggregator.
        """

        super().__init__(**kwargs)
        self._reduce_params = dict([(ReduceOp.SUM, []),])
        self._gather_params = []
        self.client_indexes = client_indexes
        # self.rank = rank
        # self.group = group

    def add_reduce_param(self, name, param, op=ReduceOp.SUM):
        """
        Add a parameter to be reduced.

        Args:
            name (str): The name of the parameter.
            param (torch.Tensor): The parameter to be reduced.
            op (ReduceOp, optional): The reduction operation (default is ReduceOp.SUM).

        Returns:
            None
        """
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
        Add parameters to be gathered from clients.

        Args:
            client_index (int): The client index for which the parameter is added.
            name (str): The name of the parameter.
            param (torch.Tensor): The parameter to be gathered.

        Returns:
            None
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
        """
        Perform communication between local aggregator and server.

        Args:
            rank (int): The rank of the local aggregator.
            groups (dict): Dictionary of communication groups.
            client_schedule (list, optional): Schedule of client indexes (default is None).

        Returns:
            None
        """
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
    """
    Parameters sent from a client to a local aggregator for aggregation.

    This class represents parameters that are sent from a client to a local aggregator
    for aggregation and communication within a local group.

    Attributes:
        client_index (int): The client index.
        _reduce_params (dict): A dictionary containing lists of parameters to be reduced using different operations.
        _gather_params (list): A list of parameter names to be gathered by the local aggregator.
    """
    def __init__(self, client_index, **kwargs):
        """
        Initialize the ClientToLocalAggregatorParams object.

        Args:
            client_index (int): The client index for which the parameters are intended.
        """
        super().__init__(**kwargs)
        self.client_index = client_index
        self._reduce_params = dict([(ReduceOp.MEAN, []), (ReduceOp.SUM, []),])
        self._gather_params = []

    def add_reduce_param(self, name, param, op=ReduceOp.SUM):
        """
        Add a parameter to be reduced.

        Args:
            name (str): The name of the parameter.
            param (torch.Tensor): The parameter to be reduced.
            op (ReduceOp, optional): The reduction operation (default is ReduceOp.SUM).

        Returns:
            None
        """
        self.__dict__.update({name: param})
        self._reduce_params[op].append(name)

    def add_gather_params(self, name, param):
        """
        Add parameters to be gathered by the local aggregator.

        Args:
            name (str): The name of the parameter.
            param (torch.Tensor): The parameter to be gathered.

        Returns:
            None
        """
        self.__dict__.update({name: param})
        self._gather_params.append(name)

    def get_mean_reduce_param_names(self):
        """
        Get the names of parameters to be reduced with the MEAN operation.

        Returns:
            list: A list of parameter names.
        """
        return self._reduce_params[ReduceOp.MEAN]

    def get_sum_reduce_param_names(self):
        """
        Get the names of parameters to be reduced with the SUM operation.

        Returns:
            list: A list of parameter names.
        """
        return self._reduce_params[ReduceOp.SUM]

    def get_gather_param_names(self):
        """
        Get the names of parameters to be gathered by the local aggregator.

        Returns:
            list: A list of parameter names.
        """
        return self._gather_params


def local_gather(local_gather_params):
    """
    Perform local gathering of parameters.

    Args:
        local_gather_params (ClientToLocalAggregatorParams): Parameters to be gathered.

    Returns:
        None
    """
    pass
