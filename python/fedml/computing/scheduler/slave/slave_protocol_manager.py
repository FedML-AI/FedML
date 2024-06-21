
import os
from ..comm_utils.job_cleanup import JobCleanup
from .base_slave_protocol_manager import FedMLBaseSlaveProtocolManager
from .launch_job_runner_manager import FedMLLaunchJobRunnerManager


class FedMLLaunchSlaveProtocolManager(FedMLBaseSlaveProtocolManager):

    def __init__(self, args, agent_config=None):
        FedMLBaseSlaveProtocolManager.__init__(self, args, agent_config=agent_config)
        self.message_center_name = "launch_slave_agent"

    # Override
    def generate_topics(self):
        super().generate_topics()

    # Override
    def add_protocol_handler(self):
        super().add_protocol_handler()

    # Override
    def _generate_protocol_manager_instance(self, args, agent_config=None):
        return FedMLLaunchSlaveProtocolManager(args, agent_config=agent_config)

    # Override
    def _get_job_runner_manager(self):
        return FedMLLaunchJobRunnerManager.get_instance()

    # Override
    def _process_connection_ready(self):
        from fedml.core.mlops import sync_deploy_id
        sync_deploy_id(
            self.edge_id, self.model_device_server_id, self.model_device_client_edge_id_list,
            message_center=self.message_center)

    # Override
    def _process_connection_lost(self):
        pass

    # Override
    def _init_extra_items(self):
        super()._init_extra_items()

        # Sync the data when startup
        JobCleanup.get_instance().sync_data_on_startup(self.args.edge_id)

        # Start the monitor process
        self.mlops_metrics.stop_device_realtime_perf()
        self.mlops_metrics.report_device_realtime_perf(self.args, self.args.agent_config["mqtt_config"])

    def save_deploy_ids(self, deploy_master_edge_id=None, deploy_slave_edge_id=None):
        if deploy_master_edge_id is not None:
            self.model_device_server_id = deploy_master_edge_id

        if deploy_slave_edge_id is not None:
            if self.model_device_client_edge_id_list is None:
                self.model_device_client_edge_id_list = list()
            self.model_device_client_edge_id_list.append(deploy_slave_edge_id)

        # Save the deployed master and worker id list to the environment variable.
        os.environ["FEDML_DEPLOY_MASTER_ID"] = str(self.model_device_server_id)
        os.environ["FEDML_DEPLOY_WORKER_IDS"] = str(self.model_device_client_edge_id_list)
