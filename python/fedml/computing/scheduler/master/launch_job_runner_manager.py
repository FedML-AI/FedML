
from fedml.core.common.singleton import Singleton
from .launch_job_runner import FedMLLaunchMasterJobRunner
from .base_master_job_runner_manager import FedMLBaseMasterJobRunnerManager


class FedMLLaunchJobRunnerManager(FedMLBaseMasterJobRunnerManager, Singleton):
    def __init__(self):
        FedMLBaseMasterJobRunnerManager.__init__(self)

    @staticmethod
    def get_instance():
        return FedMLLaunchJobRunnerManager()

    # Override
    def _generate_job_runner_instance(
            self, args, run_id=None, request_json=None, agent_config=None, edge_id=None
    ):
        return FedMLLaunchMasterJobRunner(
            args, run_id=run_id, request_json=request_json, agent_config=agent_config, edge_id=edge_id)
