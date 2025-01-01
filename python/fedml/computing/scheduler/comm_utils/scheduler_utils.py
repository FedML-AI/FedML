import os
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants


class SchedulerUtils:
    @staticmethod
    def is_using_k8s() -> bool:
        """Check if using k8s scheduler for model deploy"""
        scheduler_env = os.getenv(SchedulerConstants.SCHEDULER_TYPE_PARAM_NAME, "").upper()
        return scheduler_env == SchedulerConstants.SCHEDULER_TYPE_K8S