import subprocess
from typing import List

from GPUtil import GPUtil, GPU

from fedml.computing.scheduler.comm_utils.GPUCardUtil import GPUCard, GPUCardUtil, GPUCardType


def _convert(gpu: GPU) -> GPUCard:
    return GPUCard(
        id=gpu.id,
        uuid=gpu.uuid,
        name=gpu.name,
        load=gpu.load,
        memoryTotal=gpu.memoryTotal,
        memoryUsed=gpu.memoryUsed,
        memoryFree=gpu.memoryFree,
        driver=gpu.driver,
        serial=gpu.serial,
        display_mode=gpu.display_mode,
        display_active=gpu.display_active,
        temperature=gpu.temperature
    )


class NvidiaGPUtil(GPUCardUtil):

    @staticmethod
    def getAvailableGPUCardIDs() -> List[int]:
        return GPUtil.getAvailable()

    @staticmethod
    def getGPUCards() -> List[GPUCard]:
        return [_convert(gpu) for gpu in GPUtil.getGPUs()]

    @classmethod
    def detectGPUCardType(cls):
        try:
            subprocess.check_output(["nvidia-smi"], universal_newlines=True)
            return GPUCardType.NVIDIA
        except Exception:
            return None
