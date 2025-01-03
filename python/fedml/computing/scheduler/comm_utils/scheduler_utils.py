import os
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants


class SchedulerUtils:
    @staticmethod
    def is_using_k8s() -> bool:
        """Check if using k8s scheduler for model deploy"""
        scheduler_env = os.getenv(SchedulerConstants.SCHEDULER_TYPE_PARAM_NAME, "").upper()
        return scheduler_env == SchedulerConstants.SCHEDULER_TYPE_K8S
    
    @staticmethod
    def get_replicate_num_per_pod() -> int:
        return SchedulerConstants.REPLICATE_NUM_PER_POD
    
    @staticmethod
    def get_current_model_dir() -> str:
        return os.path.join(ClientConstants.get_model_package_dir(), SchedulerConstants.CURRENT_MODEL_DIR)

    @staticmethod
    def get_current_model_ready_file() -> str:
        current_model_dir = SchedulerUtils.get_current_model_dir()
        ready_file = os.path.join(current_model_dir, SchedulerConstants.K8S_POD_READY_FILE)
        return ready_file

    @staticmethod
    def get_pod_name_file() -> str:
        current_model_dir = SchedulerUtils.get_current_model_dir()
        pod_name_file = os.path.join(current_model_dir, SchedulerConstants.K8S_POD_NAME_FILE)
        return pod_name_file