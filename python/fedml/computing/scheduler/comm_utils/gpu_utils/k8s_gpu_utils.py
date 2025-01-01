import logging
from typing import List, Optional, Dict

from docker import DockerClient
from fedml.computing.scheduler.comm_utils.gpu_utils.gpu_utils import GPUCard, GPUCardUtil, GPUCardType
from fedml.computing.scheduler.comm_utils.crictl_utils import CriClient


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
            cri_client = CriClient()
            # Get container IDs for containers with name containing "container-task"
            container_ids = cri_client._run_command(
                ["crictl", "ps", "-q", "--name", "container-task"]
            ).strip().split('\n')
            
            gpu_cards = set()  # Use set to avoid duplicates
            gpu_index = 0  # Initialize counter for auto-incrementing GPU IDs

            for container_id in container_ids:
                if not container_id:  # Skip empty strings
                    continue
                    
                gpu_info = cri_client.get_gpu_info(container_id)
                if gpu_info and gpu_info['gpu_ids']:
                    for gpu_id in gpu_info['gpu_ids']:
                        gpu_cards.add(GPUCard(
                            id=gpu_index,
                            name=str(gpu_id),
                            driver="",
                            serial="",
                            vendor="",
                            memoryTotal=0,
                            memoryFree=0,
                            memoryUsed=0,
                            memoryUtil=0,
                            load=0,
                            device_path="",
                            uuid=str(gpu_id),
                            display_mode="",
                            display_active="",
                            temperature=0
                        ))
                        gpu_index += 1
            print(f"[K8sGPUtil] get_gpu_cards Detected {len(gpu_cards)} GPU cards, gpu_cards: {gpu_cards}")
            return list(gpu_cards)
            
        except Exception as e:
            logging.error(f"[K8sGPUtil] get_gpu_cards Failed to get GPU cards: {e}")
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