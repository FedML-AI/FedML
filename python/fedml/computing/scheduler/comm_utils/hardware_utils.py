import logging
from typing import Optional, List, Dict

from docker import DockerClient

from fedml.computing.scheduler.comm_utils.gpu_utils.gpu_utils import GPUCardUtil, GPUCard
from fedml.computing.scheduler.comm_utils.gpu_utils.nvidia_utils import NvidiaGPUtil
from fedml.computing.scheduler.comm_utils.gpu_utils.qualcomm_utils import QualcommNPUtil
from fedml.computing.scheduler.comm_utils.singleton import Singleton

GPU_CARD_UTILS = [NvidiaGPUtil, QualcommNPUtil]


class HardwareUtil(metaclass=Singleton):
    __gpu_util: Optional[GPUCardUtil] = None

    @classmethod
    def __get_util(cls) -> Optional[GPUCardUtil]:
        if cls.__gpu_util is not None:
            return cls.__gpu_util

        for gpu_util in GPU_CARD_UTILS:
            try:
                if gpu_util.detect_gpu_card_type() is not None:
                    cls.__gpu_util = gpu_util()
                    return cls.__gpu_util
            except Exception as e:
                pass

        # logging.error("No GPU card detected")
        return None

    @staticmethod
    def get_gpus() -> List[GPUCard]:
        gpu_util = HardwareUtil.__get_util()
        return gpu_util.get_gpu_cards() if gpu_util is not None else []

    @staticmethod
    def get_available_gpu_ids(order: str = "memory", limit: int = 1, max_load: float = 0.01,
                              max_memory: float = 0.01) -> List[int]:
        gpu_util = HardwareUtil.__get_util()
        return gpu_util.get_available_gpu_card_ids(order, limit, max_load, max_memory) if gpu_util is not None else []

    @staticmethod
    def get_docker_gpu_device_mapping(gpu_ids: Optional[List[int]], num_gpus: int = 0) -> Optional[Dict]:
        gpu_util = HardwareUtil.__get_util()
        if gpu_util is not None:
            return gpu_util.get_docker_gpu_device_mapping(gpu_ids, num_gpus)
        return None

    @staticmethod
    def get_docker_gpu_ids_by_container_name(container_name: str, docker_client: DockerClient) -> List[int]:
        gpu_ids = []
        gpu_util = HardwareUtil.__get_util()
        if gpu_util is not None:
            gpu_ids = gpu_util.get_docker_gpu_ids_by_container_name(container_name, docker_client)
        return gpu_ids


if __name__ == "__main__":
    gpus = HardwareUtil.get_gpus()
    get_available_gpu_cards = HardwareUtil.get_available_gpu_ids(limit=len(gpus))
    print(gpus)
    print(get_available_gpu_cards)
