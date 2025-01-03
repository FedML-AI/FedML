import logging
from typing import List, Optional, Dict

from docker import DockerClient
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.comm_utils.gpu_utils.gpu_utils import GPUCard, GPUCardUtil, GPUCardType
from fedml.computing.scheduler.comm_utils.crictl_utils import CriClient
from fedml.computing.scheduler.comm_utils.scheduler_utils import SchedulerUtils
from fedml.computing.scheduler.slave.client_constants import ClientConstants


class K8sGPUtil(GPUCardUtil):
    @classmethod
    def detect_gpu_card_type(cls) -> Optional[GPUCardType]:
        try:
            # Check if we're running in a Kubernetes environment
            return GPUCardType.K8SGPU  # Assuming K8s is using NVIDIA GPUs
        except Exception:
            return None

    @staticmethod
    def get_gpu_cards() -> List[GPUCard]:
        try:
            import json
            import os

            # Path to GPU info file as defined in the k8s deployment
            current_model_dir = SchedulerUtils.get_current_model_dir()
            gpu_info_path = os.path.join(current_model_dir, SchedulerConstants.K8S_POD_GPU_INFO_FILE)
            if not os.path.exists(gpu_info_path):
                logging.warning(f"[K8sGPUtil] GPU info file not found at {gpu_info_path}")
                return []

            with open(gpu_info_path, 'r') as f:
                gpu_data = json.load(f)

            gpu_cards = []
            for gpu_info in gpu_data:
                card = GPUCard(
                    id=gpu_info["id"],
                    name=gpu_info["name"],
                    driver=gpu_info["driver"],
                    serial=gpu_info["serial"],
                    vendor=gpu_info["vendor"],
                    memoryTotal=gpu_info["memoryTotal"],
                    memoryFree=gpu_info["memoryFree"],
                    memoryUsed=gpu_info["memoryUsed"],
                    memoryUtil=gpu_info["memoryUtil"],
                    load=gpu_info["load"],
                    uuid=gpu_info["uuid"],
                    display_mode=gpu_info["display_mode"],
                    display_active=gpu_info["display_active"],
                    temperature=gpu_info["temperature"]
                )
                gpu_cards.append(card)
            logging.info(f"[K8sGPUtil] get_gpu_cards gpu_cards: {gpu_cards}")
            return gpu_cards

        except Exception as e:
            logging.error(f"[K8sGPUtil] Error reading GPU information: {str(e)}")
            return []

    @staticmethod
    def get_available_gpu_card_ids(order: str, limit: int, max_load: float, max_memory: float) -> List[int]:
        gpu_cards = K8sGPUtil.get_gpu_cards()
        gpu_ids = [card.id for card in gpu_cards]
        logging.info(f"[K8sGPUtil] get_available_gpu_card_ids gpu_ids: {gpu_ids}")
        return gpu_ids

    @staticmethod
    def get_docker_gpu_device_mapping(gpu_ids: List[int], num_gpus: int = 0) -> Optional[Dict]:
        # K8s handles GPU allocation differently, so this might not be needed
        return None

    @staticmethod
    def get_docker_gpu_ids_by_container_name(container_name: str, docker_client: DockerClient) -> List[int]:
        gpu_cards = K8sGPUtil.get_gpu_cards()
        gpu_ids = [card.id for card in gpu_cards]
        logging.info(f"[K8sGPUtil] get_docker_gpu_ids_by_container_name gpu_ids: {gpu_ids}")
        return gpu_ids