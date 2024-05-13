
from abc import ABC, abstractmethod
from ..scheduler_core.scheduler_base_job_runner_manager import FedMLSchedulerBaseJobRunnerManager
from ..scheduler_core.scheduler_base_job_runner import FedMLSchedulerBaseJobRunner


class FedMLBaseSlaveJobRunnerManager(FedMLSchedulerBaseJobRunnerManager, ABC):
    def __init__(self):
        FedMLSchedulerBaseJobRunnerManager.__init__(self)

    def cleanup_containers_and_release_gpus(self, run_id, edge_id, job_type):
        FedMLSchedulerBaseJobRunner.cleanup_containers_and_release_gpus(run_id, edge_id, job_type)
