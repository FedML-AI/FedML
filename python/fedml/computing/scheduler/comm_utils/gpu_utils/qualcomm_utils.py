import logging
import subprocess
from typing import List, Optional

from fedml.computing.scheduler.comm_utils.gpu_utils.gpu_utils import GPUCard, GPUCardUtil, GPUCardType
from qaicrt import Util, QIDList, QDevInfo, QStatus


class QualcommNPUtil(GPUCardUtil):
    @classmethod
    def detect_gpu_card_type(cls) -> Optional[GPUCardType]:
        try:
            subprocess.check_output(["/opt/qti-aic/tools/qaic-util"], universal_newlines=True)
            return GPUCardType.QUALCOMM
        except Exception:
            return None

    @staticmethod
    def get_gpu_cards() -> List[GPUCard]:
        cards = []
        util = Util()
        status, card_list = util.getDeviceIds()
        if status.value == 0:
            for card in card_list:
                status, card_info = util.getDeviceInfo(card)
                if status.value == 0 and card_info.devStatus.value == 1:
                    cards.append(QualcommNPUtil.__convert(card_info))

        else:
            logging.error("Qualcomm Card Status not Healthy")
        return cards

    @staticmethod
    def get_available_gpu_card_ids(order: str = "memory", limit: int = 1, max_load: float = 0.01,
                                   max_memory: float = 0.01) -> List[int]:
        available_gpu_card_ids = []

        if order != "memory":
            raise NotImplementedError(f"Qualcomm utils doesn't have support to compute availability based on {order}. "
                                      f"Supported criteria: [memory]")

        return available_gpu_card_ids

    @staticmethod
    def __convert(npu) -> GPUCard:
        memory_total = npu.devData.resourceInfo.dramTotal / 1024
        memory_free = npu.devData.resourceInfo.dramFree / 1024
        memory_used = memory_total - memory_free
        memory_utilized = float(memory_used) / float(memory_total)

        return GPUCard(
            id=npu.qid,
            name=npu.pciInfo.devicename,
            driver=npu.devData.fwQCImageVersionString,
            serial=npu.devData.serial,
            memoryTotal=memory_total,
            memoryFree=memory_free,
            memoryUsed=memory_used,
            memoryUtil=memory_utilized,
        )
