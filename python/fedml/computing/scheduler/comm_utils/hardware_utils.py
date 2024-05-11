import logging
from typing import Optional, List

from fedml.computing.scheduler.comm_utils.gpu_utils import GPUCardUtil, GPUCard
from fedml.computing.scheduler.comm_utils.singleton import Singleton


class HardwareUtil(metaclass=Singleton):

    def __init__(self):
        self._gpu_util: Optional[GPUCardUtil] = self.__get_util()

    @staticmethod
    def __get_util() -> Optional[GPUCardUtil]:
        for cls in GPUCardUtil.__subclasses__():
            try:
                if cls.detectGPUCardType() is not None:
                    return cls()
            except Exception as e:
                pass

        logging.error("No GPU card detected")
        return None

    def getGPUs(self) -> List[GPUCard]:
        if self._gpu_util is None:
            return []
        return self._gpu_util.getGPUCards()

    def getAvailableGPUCardIDs(self) -> List[int]:
        if self._gpu_util is None:
            return []
        return self._gpu_util.getAvailableGPUCardIDs()

