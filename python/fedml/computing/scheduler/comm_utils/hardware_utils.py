import logging

from typing import Optional, List

from fedml.computing.scheduler.comm_utils.gpu_utils.gpu_utils import GPUCardUtil, GPUCard
from fedml.computing.scheduler.comm_utils.gpu_utils.nvidia_utils import NvidiaGPUtil
from fedml.computing.scheduler.comm_utils.singleton import Singleton


class HardwareUtil(metaclass=Singleton):

    _gpu_utils = [NvidiaGPUtil]
    _gpu_util: Optional[GPUCardUtil] = None

    @staticmethod
    def _get_util() -> Optional[GPUCardUtil]:
        if HardwareUtil._gpu_util is not None:
            return HardwareUtil._gpu_util

        for gpu_util in HardwareUtil._gpu_utils:
            try:
                if gpu_util.detectGPUCardType() is not None:
                    HardwareUtil._gpu_util = gpu_util()
                    return HardwareUtil._gpu_util
            except Exception as e:
                pass

        logging.error("No GPU card detected")
        return None

    @staticmethod
    def getGPUs() -> List[GPUCard]:
        gpu_util = HardwareUtil._get_util()
        return gpu_util.getGPUCards() if gpu_util is not None else []

    @staticmethod
    def getAvailableGPUCardIDs() -> List[int]:
        gpu_util = HardwareUtil._get_util()
        return gpu_util.getAvailainfbleGPUCardIDs() if gpu_util is not None else []


if __name__ == "__main__":
    gpus = HardwareUtil.getGPUs()
    get_available_gpu_cards = HardwareUtil.getAvailableGPUCardIDs()
    print(gpus)
    print(get_available_gpu_cards)
