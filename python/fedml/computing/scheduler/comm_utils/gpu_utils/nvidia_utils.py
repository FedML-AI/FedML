import subprocess
from typing import List, Optional

from GPUtil import GPUtil, GPU

from fedml.computing.scheduler.comm_utils.gpu_utils.gpu_utils import GPUCard, GPUCardUtil, GPUCardType


class NvidiaGPUtil(GPUCardUtil):
    @classmethod
    def detect_gpu_card_type(cls) -> Optional[GPUCardType]:
        try:
            subprocess.check_output(["nvidia-smi"], universal_newlines=True)
            return GPUCardType.NVIDIA
        except Exception:
            return None

    @staticmethod
    def get_gpu_cards() -> List[GPUCard]:
        return [NvidiaGPUtil.__convert(gpu) for gpu in GPUtil.getGPUs()]

    @staticmethod
    def get_available_gpu_card_ids(order: str, limit: int, max_load: float, max_memory: float) -> List[int]:
        return GPUtil.getAvailable(order=order, limit=limit, maxLoad=max_load, maxMemory=max_memory)

    @staticmethod
    def __convert(gpu: GPU) -> GPUCard:
        return GPUCard(
            id=gpu.id,
            name=gpu.name,
            driver=gpu.driver,
            serial=gpu.serial,
            vendor=GPUCardType.NVIDIA.name,
            memoryTotal=gpu.memoryTotal,
            memoryFree=gpu.memoryFree,
            memoryUsed=gpu.memoryUsed,
            memoryUtil=gpu.memoryUtil,
            load=gpu.load,
            uuid=gpu.uuid,
            display_mode=gpu.display_mode,
            display_active=gpu.display_active,
            temperature=gpu.temperature,
        )
