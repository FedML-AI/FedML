import logging


WORKER_NAME = "worker{}"

# Generate Device Map for Cuda RPC
def set_device_map(options, worker_idx, device_list):
    """
    Set the device mapping for PyTorch RPC communication between workers.

    Args:
        options (rpc.TensorPipeRpcBackendOptions): The RPC backend options to configure.
        worker_idx (int): The index of the current worker.
        device_list (list of str): A list of device identifiers for all workers.

    Example:
        Suppose you have two workers with GPUs, and `device_list` is ['cuda:0', 'cuda:1'].
        If `worker_idx` is 0, this function will set the device mapping for worker 0 as follows:
        {WORKER_NAME.format(1): 'cuda:1'} to communicate with worker 1 using 'cuda:1'.

    Returns:
        None
    """
    local_device = device_list[worker_idx]
    for index, remote_device in enumerate(device_list):
        logging.warn(
            f"Setting device map for client {index} as {remote_device}")
        if index != worker_idx:
            options.set_device_map(WORKER_NAME.format(
                index), {local_device: remote_device})
