import logging
import torch.distributed as dist
import os


class ProcessGroupManager:
    def __init__(self, rank, world_size, master_address, master_port, only_gpu):
        logging.info("Start process group")
        logging.info(
            "rank: %d, world_size: %d, master_address: %s, master_port: %d"
            % (rank, world_size, master_address, master_port)
        )
        os.environ["MASTER_ADDR"] = master_address
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

        env_dict = {
            key: os.environ[key]
            for key in (
                "MASTER_ADDR",
                "MASTER_PORT",
                "RANK",
                "WORLD_SIZE",
                "NCCL_SOCKET_IFNAME",
            )
        }
        logging.info(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        backend = dist.Backend.NCCL if only_gpu else dist.Backend.GLOO
        # initialize the process group
        dist.init_process_group(
            backend=backend,
            init_method="tcp://" + master_address + ":" + str(master_port),
            rank=rank,
            world_size=world_size,
        )
        self.messaging_pg = dist.new_group()
        # dist.init_process_group()

        logging.info("Initiated")

    def cleanup(self):
        dist.destroy_process_group()

    def get_process_group(self):
        return self.messaging_pg
