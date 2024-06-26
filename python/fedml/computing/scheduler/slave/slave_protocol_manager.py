
import copy
import json
import os
import uuid

from fedml.computing.scheduler.scheduler_core.general_constants import GeneralConstants
from fedml.core.distributed.communication.mqtt.mqtt_manager import MqttManager

from ..comm_utils.job_cleanup import JobCleanup
from .base_slave_protocol_manager import FedMLBaseSlaveProtocolManager
from .launch_job_runner_manager import FedMLLaunchJobRunnerManager


class FedMLLaunchSlaveProtocolManager(FedMLBaseSlaveProtocolManager):

    def __init__(self, args, agent_config=None):
        FedMLBaseSlaveProtocolManager.__init__(self, args, agent_config=agent_config)
        self.message_center_name = "launch_slave_agent"

    def generate_communication_manager(self):
        if self.communication_mgr is None:
            self.communication_mgr = MqttManager(
                self.agent_config["mqtt_config"]["BROKER_HOST"],
                self.agent_config["mqtt_config"]["BROKER_PORT"],
                self.agent_config["mqtt_config"]["MQTT_USER"],
                self.agent_config["mqtt_config"]["MQTT_PWD"],
                self.agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
                f"FedML_Launch_Slave_Agent_@{self.user_name}@_@{self.current_device_id}@_@{str(uuid.uuid4())}@",
                self.topic_last_will,
                json.dumps({"ID": self.edge_id, "status": GeneralConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE})
            )

    # Override
    def _get_job_runner_manager(self):
        return FedMLLaunchJobRunnerManager.get_instance()

    # Override
    def _process_connection_ready(self):
        from fedml.core.mlops import sync_deploy_id
        sync_deploy_id(
            self.edge_id, self.model_device_server_id, self.model_device_client_edge_id_list,
            message_center=self.message_center)

    def _process_connection_lost(self):
        pass

    # TODO(alaydshah): Double initialization of deploy master and worker agents. This can be potentially removed.
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
