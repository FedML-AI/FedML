from abc import ABC

from .base_master_protocol_manager import FedMLBaseMasterProtocolManager
from .launch_job_runner_manager import FedMLLaunchJobRunnerManager


class FedMLLaunchMasterProtocolManager(FedMLBaseMasterProtocolManager, ABC):
    def __init__(self, args, agent_config=None):
        FedMLBaseMasterProtocolManager.__init__(self, args, agent_config=agent_config)

    # Override
    def generate_topics(self):
        super().generate_topics()

    # Override
    def add_protocol_handler(self):
        super().add_protocol_handler()

    # Override
    def _generate_protocol_manager_instance(self, args, agent_config=None):
        return FedMLLaunchMasterProtocolManager(args, agent_config=agent_config)

    # Override
    def _get_job_runner_manager(self):
        return FedMLLaunchJobRunnerManager.get_instance()

    # Override
    def _init_extra_items(self):
        # Start the monitor process
        self.mlops_metrics.stop_device_realtime_perf()
        self.mlops_metrics.report_device_realtime_perf(
            self.args, self.args.agent_config["mqtt_config"], is_client=False)

    # Override
    def print_connected_info(self):
        super().print_connected_info()

    # Override
    def _process_job_complete_status(self, run_id, server_id, complete_payload):
        # Complete the job runner
        self._get_job_runner_manager().complete_job_runner(
            run_id, args=self.args, server_id=server_id, request_json=complete_payload,
            run_as_cloud_agent=self.run_as_cloud_agent, run_as_cloud_server=self.run_as_cloud_server,
            use_local_process_as_cloud_server=self.use_local_process_as_cloud_server)
