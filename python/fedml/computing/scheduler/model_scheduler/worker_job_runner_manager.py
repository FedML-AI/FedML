
from fedml.core.common.singleton import Singleton
from .worker_job_runner import FedMLDeployWorkerJobRunner
from ..scheduler_core.general_constants import GeneralConstants
from ..slave.base_slave_job_runner_manager import FedMLBaseSlaveJobRunnerManager


class FedMLDeployJobRunnerManager(FedMLBaseSlaveJobRunnerManager, Singleton):
    def __init__(self):
        FedMLBaseSlaveJobRunnerManager.__init__(self)

    @staticmethod
    def get_instance():
        return FedMLDeployJobRunnerManager()

    # Override
    def _generate_job_runner_instance(
            self, args, run_id=None, request_json=None, agent_config=None, edge_id=None
    ):
        job_runner = FedMLDeployWorkerJobRunner(
            args, run_id=run_id, request_json=request_json, agent_config=agent_config, edge_id=edge_id)
        job_runner.infer_host = GeneralConstants.get_ip_address(request_json)
        return job_runner
