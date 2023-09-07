import os
from enum import Enum

from enum import Enum

import torch
import torch.distributed as dist

""" model util """


def get_weights(state):
    """
    Returns a list of weights from the state dictionary.

    Args:
        state (dict): The state dictionary containing model parameters.

    Returns:
        list or None: A list of model weights or None if the state is None.
    """"
    if state is not None:
        return list(state.values())
    else:
        return None


def set_model_params_with_list(model, new_model_params):
    """
    Set the model parameters with a list of new parameters.

    Args:
        model: The model whose parameters will be updated.
        new_model_params (list): A list of new model parameters.
    """
    for model_param, model_update_param in zip(model.parameters(), new_model_params):
        print(f"model_param.shape: {model_param.shape}, model_update_param.shape: {model_update_param.shape}")
        # model_param.data = model_update_param


def clear_optim_buffer(optimizer):
    """
    Clear the optimization buffer for momentum.

    Args:
        optimizer: The optimizer whose buffer will be cleared.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            param_state = optimizer.state[p]
            # Reinitialize momentum buffer
            if "momentum_buffer" in param_state:
                param_state["momentum_buffer"].zero_()


""" cpu --- gpu """


def optimizer_to(optim, device):
    """
    Move optimizer parameters to the specified device.

    Args:
        optim: The optimizer whose parameters will be moved.
        device (str): The target device (e.g., 'cpu' or 'cuda').
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def move_to_cpu(model, optimizer):
    """
    Move the model and optimizer to the CPU.

    Args:
        model: The model to be moved.
        optimizer: The optimizer to be moved.

    Returns:
        model: The model on the CPU.
    """
    if str(next(model.parameters()).device) == "cpu":
        pass
    else:
        model = model.to("cpu")
        # optimizer_to(self.trainer.optimizer, 'cpu')
    if len(list(optimizer.state.values())) > 0:
        optimizer_to(optimizer, "cpu")
    return model


def move_to_gpu(model, optimizer, device):
    """
    Move the model and optimizer to the specified GPU device.

    Args:
        model: The model to be moved.
        optimizer: The optimizer to be moved.
        device (str): The target GPU device (e.g., 'cuda').

    Returns:
        model: The model on the specified GPU device.
    """
    if str(next(model.parameters()).device) == "cpu":
        model = model.to(device)
    else:
        pass

    # logging.info(self.trainer.optimizer.state.values())
    if len(list(optimizer.state.values())) > 0:
        optimizer_to(optimizer, device)
    return model


""" communication """


class ReduceOp(Enum):
    """
    Reduction Op to perform in reduce/allreduce ops. The reduction op is applied element-wise.
    """

    SUM = 1
    MEAN = 2


class Role(Enum):
    SERVER = 1
    DEVICE = 2


class CommState:
    role: Role = None
    server_rank = -1
    server_size = -1
    device_size = -1
    process_id = -1
    device_id = -1
    # comm: dist.Communicator = None
    # role_comm: dist.Communicator = None


def init_ddp(args):
    """
    Initialize Distributed Data Parallel (DDP) for training.

    Args:
        args: The arguments containing DDP configuration.

    Returns:
        global_rank (int): The global rank of the current process.
        world_size (int): The total number of processes in the world.
    """ 
    # use InfiniBand
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    # This the global rank: 0, 1, 2, ..., 15
    global_rank = int(os.environ["RANK"])
    print("int(os.environ['RANK']) = %d" % global_rank)

    # This the globak world_size
    world_size = int(os.environ["WORLD_SIZE"])
    print("world_size = %d" % world_size)

    # initialize the process group
    # dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    # dist.init_process_group(backend="nccl", init_method="env://")
    dist.init_process_group(backend="gloo", init_method="env://")

    local_rank = args.local_rank
    print(f"Running basic DDP example on local rank {local_rank}.")
    return global_rank, world_size


def FedML_NCCL_Similulation_init(args):
    """
    Initialize NCCL-based simulation environment.

    Args:
        args: The arguments containing simulation configuration.

    Returns:
        args (object): The updated arguments.
    """
    # dist.init_process_group(
    #     init_method='tcp://10.1.1.20:23456',
    #     rank=args.rank,
    #     world_size=4)

    global_rank, world_size = init_ddp(args)
    CommState.server_rank = 0
    CommState.server_size = 1
    CommState.device_size = world_size - 1
    CommState.process_id = global_rank
    # CommState.device_id = process_id - 1 if process_id > 0 else -1
    CommState.device_id = global_rank - 1
    CommState.role = Role.SERVER if global_rank == 0 else Role.DEVICE
    args.comm = CommState
    args.process_id = global_rank
    args.worker_num = world_size
    return args


# def FedML_NCCL_init(args):
#     # comm = MPI.COMM_WORLD
#     # process_id = comm.Get_rank()
#     # worker_number = comm.Get_size()
#     comm, global_rank, world_size = FedML_NCCL_Similulation_init(args)
#     process_id = global_rank
#     worker_number = world_size - 1
#     return comm, process_id, worker_number


def get_rank():
    """
    Get the rank of the current process in the distributed environment.

    Returns:
        int: The rank of the current process.
    """
    return dist.get_rank()

def get_server_rank():
    """
    Get the rank of the server process in the distributed environment.

    Returns:
        int: The rank of the server process.
    """
    return CommState.server_rank

def get_world_size():
    """
    Get the total number of processes in the distributed environment.

    Returns:
        int: The total number of processes.
    """
    return dist.get_world_size()

def get_worker_number():
    """
    Get the number of worker processes (excluding the server) in the distributed environment.

    Returns:
        int: The number of worker processes.
    """
    return CommState.device_size


def new_group(ranks):
    """
    Create a new process group with the specified ranks.

    Args:
        ranks (list): A list of ranks to include in the new group.

    Returns:
        dist.ProcessGroup: The new process group.
    """
    return dist.new_group(ranks=ranks)
    # dist.new_group(ranks=None, timeout=datetime.timedelta(seconds=1800), backend=None, pg_options=None)


def fedml_nccl_send_to_server(tensor, src=0, group=None):
    """
    Send a tensor from a device (GPU) to the server process.

    Args:
        tensor (torch.Tensor): The tensor to send.
        src (int): The source rank of the sending process.
        group (dist.ProcessGroup, optional): The process group to use for communication.

    Note:
        This function is used to send tensors from a device (GPU) to the server during communication.

    Example:
        ```python
        fedml_nccl_send_to_server(my_tensor, src=1, group=my_group)
        ```

    """
    is_cuda = tensor.is_cuda
    # if not is_cuda:
    #     logging.info("Warning: Tensor is not on GPU!!!")
    # dist.gather(tensor, gather_list=gather_list, dst=dst, group=group)
    dist.broadcast(tensor=tensor, src=src, group=group)


def fedml_nccl_broadcast(tensor, src):
    """
    Broadcast a tensor from the server process to all devices (GPUs).

    Args:
        tensor (torch.Tensor): The tensor to broadcast.
        src (int): The source rank of the broadcasting process.

    Note:
        This function is used to broadcast tensors from the server to all devices during communication.

    Example:
        ```python
        fedml_nccl_broadcast(my_tensor, src=0)
        ```

    """
    is_cuda = tensor.is_cuda
    # if not is_cuda:
    #     logging.info("Warning: Tensor is not on GPU!!!")
    input_tensor = tensor
    dist.broadcast(tensor=input_tensor, src=src)


def fedml_nccl_reduce(tensor, dst, op: ReduceOp = ReduceOp.SUM):
    """
    Reduce a tensor across processes with the specified reduction operation.

    Args:
        tensor (torch.Tensor): The tensor to reduce.
        dst (int): The destination rank for the reduced tensor.
        op (ReduceOp): The reduction operation (SUM or MEAN). Currently only supports SUM and MEAN reduction ops

    Note:
        This function is used to perform reduction operations (SUM or MEAN) on tensors across processes.

    Example:
        ```python
        fedml_nccl_reduce(my_tensor, dst=0, op=ReduceOp.SUM)
        ```

    """
    is_cuda = tensor.is_cuda
    # if not is_cuda:
    #     logging.info("Warning: Tensor is not on GPU!!!")
    if get_rank() == dst:
        tensor.zero_()
    if op == ReduceOp.SUM:
        dist.reduce(tensor=tensor, dst=dst, op=dist.ReduceOp.SUM)
    elif op == ReduceOp.MEAN:
        raise NotImplementedError
    else:
        raise NotImplementedError


# def fedml_nccl_allreduce(src, op=dist.ReduceOp.SUM, grou=None):
#     dst.copy_(src)
#     dist.all_reduce(tensor, op)


def fedml_nccl_barrier():
    """
    Synchronize all processes in the distributed environment.

    Note:
        This function is used to ensure that all processes reach a barrier and synchronize their execution.

    Example:
        ```python
        fedml_nccl_barrier()
        ```

    """
    dist.barrier()


def broadcast_model_state(state_dict, src):
    """
    Broadcast the model's state dictionary from the server process to all devices (GPUs).

    Args:
        state_dict (dict): The model's state dictionary to broadcast.
        src (int): The source rank of the broadcasting process.

    Note:
        This function is used to broadcast the model's state dictionary from the server to all devices during communication.

    Example:
        ```python
        broadcast_model_state(my_state_dict, src=0)
        ```

    """
    # for name, param in state_dict.items():
    #     logging.info(f"name:{name}, param.shape: {param.shape}")
    for param in state_dict.values():
        # logging.info(f"In broadcast_model_state... param: {param}, param.shape: {param.shape}")
        # logging.info(f"In broadcast_model_state... param.shape: {param.shape}")
        dist.broadcast(tensor=param, src=src)
