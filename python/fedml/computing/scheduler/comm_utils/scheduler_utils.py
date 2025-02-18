import os
from os.path import expanduser
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants


class SchedulerUtils:
    @staticmethod
    def is_using_k8s() -> bool:
        """Check if using k8s scheduler for fedml"""
        scheduler_env = os.getenv(SchedulerConstants.SCHEDULER_TYPE_PARAM_NAME, "").upper()
        return scheduler_env == SchedulerConstants.SCHEDULER_TYPE_K8S
    
    @staticmethod
    def is_using_k8s_for_deploy() -> bool:
        """Check if using k8s scheduler and is deploy task"""
        scheduler_env = os.getenv(SchedulerConstants.SCHEDULER_TYPE_PARAM_NAME, "").upper()
        scheduler_task_env = os.getenv(SchedulerConstants.SCHEDULER_TASK_TYPE_PARAM_NAME, "").upper()
        return (scheduler_env == SchedulerConstants.SCHEDULER_TYPE_K8S and
                scheduler_task_env == SchedulerConstants.SCHEDULER_TASK_TYPE_DEPLOY)
    
    @staticmethod
    def get_model_inference_gateway_port_in_k8s() -> int:
        return os.getenv(SchedulerConstants.ENV_MODEL_INFERENCE_PORT_IN_K8S, 
                         SchedulerConstants.MODEL_INFERENCE_DEFAULT_PORT_IN_K8S)
    
    @staticmethod
    def get_replicate_num_per_pod() -> int:
        return SchedulerConstants.REPLICATE_NUM_PER_POD
    
    @staticmethod
    def get_current_model_dir() -> str:
        return os.path.join(SchedulerUtils.get_model_package_dir(), SchedulerConstants.CURRENT_MODEL_DIR)

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
    
    @staticmethod
    def get_model_package_dir():
        model_packages_dir = os.path.join(SchedulerUtils.get_fedml_home_dir(), "fedml", "model_packages")
        if not os.path.exists(model_packages_dir):
            os.makedirs(model_packages_dir, exist_ok=True)
        return model_packages_dir
    
    @staticmethod
    def get_fedml_home_dir():
        home_dir = expanduser("~")
        fedml_home_dir = os.path.join(home_dir, ".fedml", SchedulerConstants.LOCAL_HOME_RUNNER_DIR_NAME)
        if not os.path.exists(fedml_home_dir):
            os.makedirs(fedml_home_dir, exist_ok=True)
        return fedml_home_dir