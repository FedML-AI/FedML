import logging
import math
import re
import subprocess
import sys
from typing import List, Optional, Dict

from docker import DockerClient

from fedml.computing.scheduler.comm_utils.gpu_utils.gpu_utils import GPUCard, GPUCardUtil, GPUCardType


class QualcommNPUtil(GPUCardUtil):
    NPU_CARD_PATH = "/dev/accel/accel"

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
        gpu_cards: List[GPUCard] = QualcommNPUtil.get_gpu_cards()
        gpu_cards = list(filter(lambda card: (card.memoryUtil < max_memory and card.load < max_load), gpu_cards))
        if order == 'memory':
            gpu_cards.sort(key=lambda card: float('inf') if math.isnan(card.memoryUtil) else card.memoryUtil, reverse=False)
        elif order == 'load':
            gpu_cards.sort(key=lambda card: float('inf') if math.isnan(card.memoryUtil) else card.load, reverse=False)
        else:
            raise NotImplementedError(f"Qualcomm utils doesn't have support to compute availability based on {order}. "
                                      f"Supported criteria: [memory, load]")

        gpu_cards = gpu_cards[0:min(limit, len(gpu_cards))]
        return list(map(lambda card: card.id, gpu_cards))

    @staticmethod
    def get_docker_gpu_device_mapping(gpu_ids: Optional[List[int]], num_gpus: int = 0) -> Optional[Dict]:
        if gpu_ids is not None and len(gpu_ids):
            return {
                "devices": [f"{QualcommNPUtil.NPU_CARD_PATH}{gpu_id}:{QualcommNPUtil.NPU_CARD_PATH}{gpu_id}" for gpu_id
                            in gpu_ids]}
        return None

    @staticmethod
    def get_docker_gpu_ids_by_container_name(container_name: str, docker_client: DockerClient) -> List[int]:
        gpu_ids = []
        try:
            docker_inspect_info = docker_client.api.inspect_container(container_name)
            gpu_ids = QualcommNPUtil.__parse_gpu_ids(docker_inspect_info.get("HostConfig", {}))
        except Exception as e:
            logging.error(f"Failed to get GPU IDs: {e}")
            pass
        return gpu_ids

    @staticmethod
    def __convert(npu) -> GPUCard:
        # TODO (alaydshah): Add support for temperature
        memory_total = npu.devData.resourceInfo.dramTotal / 1024
        memory_free = npu.devData.resourceInfo.dramFree / 1024
        memory_used = memory_total - memory_free
        memory_utilized = float(memory_used) / float(memory_total)
        nsp_free = npu.devData.resourceInfo.nspFree
        nsp_total = npu.devData.resourceInfo.nspTotal
        load = (nsp_total - nsp_free) / nsp_total

        return GPUCard(
            id=npu.qid,
            name=npu.pciInfo.devicename,
            driver=npu.devData.fwQCImageVersionString,
            serial=npu.devData.serial,
            vendor=GPUCardType.QUALCOMM.name,
            memoryTotal=memory_total,
            memoryFree=memory_free,
            memoryUsed=memory_used,
            memoryUtil=memory_utilized,
            load=load,
        )

    @staticmethod
    def __parse_gpu_ids(host_config: dict) -> List[int]:
        devices = host_config.get('Devices', [])
        gpu_ids = []
        for device in devices:
            gpu_id = QualcommNPUtil.__extract_integer_from_host_path(device.get('PathOnHost', None))

            # Check explicitly if gpu_id is not None, as gpu_id can be 0, which is a valid value to include.
            if gpu_id is not None:
                gpu_ids.append(gpu_id)
        return gpu_ids

    @staticmethod
    def __extract_integer_from_host_path(host_path: str) -> Optional[int]:
        if not host_path:
            logging.error("Host Path is None; GPU Id extraction Failed")
            return None

        npu_card_path = QualcommNPUtil.NPU_CARD_PATH

        # Check if host_path starts with npu_card_path
        if host_path.startswith(npu_card_path):

            # Extract the numeric suffix from the host path
            suffix = host_path[len(npu_card_path):]  # Get the substring after npu_card_path
            match = re.match(r'^(\d+)', suffix)  # Use regex to match the leading integer
            if match:
                return int(match.group(1))  # Return the extracted integer
            else:
                logging.error(f"Failed to extract GPU id from Host Path {host_path}")
        else:
            logging.error(f"Host Path {host_path} doesn't start with NPU Card Path {npu_card_path}")

        # Return None if extraction fails
        return None
