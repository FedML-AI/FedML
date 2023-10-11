import logging
import os
import torch
import torch.distributed as dist

class ProcessGroupManager:
    """
    Manages the initialization and cleanup of process groups for distributed training.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes in the group.
        master_address (str): The address of the master process.
        master_port (int): The port for communication with the master process.
        only_gpu (bool): Whether to use NCCL backend if GPUs are available, otherwise use GLOO.

    Attributes:
        messaging_pg (dist.ProcessGroup): The initialized process group for messaging.

    Methods:
        cleanup():
            Cleanup and destroy the process group.
        get_process_group():
            Get the initialized process group.

    """
    def __init__(self, rank, world_size, master_address, master_port, only_gpu):
        """
        Initialize the ProcessGroupManager.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes in the group.
            master_address (str): The address of the master process.
            master_port (int): The port for communication with the master process.
            only_gpu (bool): Whether to use NCCL backend if GPUs are available, otherwise use GLOO.

        Note:
            This constructor sets up the process group and environment variables.

        Returns:
            None
        """
        logging.info("Start process group")
        logging.info(
            "rank: %d, world_size: %d, master_address: %s, master_port: %s"
            % (rank, world_size, master_address, master_port)
        )
        os.environ["MASTER_ADDR"] = master_address
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

        env_dict = {key: os.environ[key] for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE",)}
        logging.info(f"[{os.getpid()}] Initializing process group with: {env_dict}")

        backend = dist.Backend.NCCL if (only_gpu and torch.cuda.is_available()) else dist.Backend.GLOO
        logging.info(f"Process group backend: {backend}")

        # initialize the process group
        dist.init_process_group(backend=backend)

        self.messaging_pg = dist.new_group(backend=backend)

        logging.info("Initiated")

    def cleanup(self):
        """
        Cleanup and destroy the process group.

        Returns:
            None
        """
        dist.destroy_process_group()

    def get_process_group(self):
        """
        Get the initialized process group.

        Returns:
            dist.ProcessGroup: The initialized process group.
        """
        return self.messaging_pg
