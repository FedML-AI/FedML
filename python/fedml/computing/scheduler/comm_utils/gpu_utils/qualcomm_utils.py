import logging
import math
import subprocess
import sys
from typing import List, Optional

from fedml.computing.scheduler.comm_utils.gpu_utils.gpu_utils import GPUCard, GPUCardUtil, GPUCardType


class QualcommNPUtil(GPUCardUtil):
    def __init__(self):
        sys.path.append("/opt/qti-aic/dev/lib/x86_64/")

    @classmethod
    def detect_gpu_card_type(cls) -> Optional[GPUCardType]:
        try:
            subprocess.check_output(["/opt/qti-aic/tools/qaic-util"], universal_newlines=True)
            return GPUCardType.QUALCOMM
        except Exception:
            return None

    @staticmethod
    def get_gpu_cards() -> List[GPUCard]:
        from qaicrt import Util, QIDList, QDevInfo, QStatus

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
    def get_available_gpu_card_ids(order: str, limit: int, max_load: float, max_memory: float) -> List[int]:

        if order != "memory":
            raise NotImplementedError(f"Qualcomm utils doesn't have support to compute availability based on {order}. "
                                      f"Supported criteria: [memory]")

        gpu_cards: List[GPUCard] = QualcommNPUtil.get_gpu_cards()
        gpu_cards = list(filter(lambda card: card.memoryUtil < max_memory, gpu_cards))
        gpu_cards.sort(key=lambda card: float('inf') if math.isnan(card.memoryUtil) else card.memoryUtil, reverse=False)
        gpu_cards = gpu_cards[0:min(limit, len(gpu_cards))]
        return list(map(lambda card: card.id, gpu_cards))

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
