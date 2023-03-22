import logging


WORKER_NAME = "worker{}"

# Generate Device Map for Cuda RPC
def set_device_map(options, worker_idx, device_list):
    local_device = device_list[worker_idx]
    for index, remote_device in enumerate(device_list):
        logging.warn(f"Setting device map for client {index} as {remote_device}")
        if index != worker_idx:
            options.set_device_map(WORKER_NAME.format(index), {local_device: remote_device})