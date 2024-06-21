import json
import uuid

from fedml.computing.scheduler.scheduler_core.general_constants import GeneralConstants
from fedml.core.distributed.communication.mqtt.mqtt_manager import MqttManager

from .base_master_protocol_manager import FedMLBaseMasterProtocolManager
from .launch_job_runner_manager import FedMLLaunchJobRunnerManager


class FedMLLaunchMasterProtocolManager(FedMLBaseMasterProtocolManager):

    def __init__(self, args, agent_config=None):
        FedMLBaseMasterProtocolManager.__init__(self, args, agent_config=agent_config)
        self.message_center_name = "launch_master_agent"

    def generate_communication_manager(self):
        if self.communication_mgr is None:
            self.communication_mgr = MqttManager(
                self.agent_config["mqtt_config"]["BROKER_HOST"],
                self.agent_config["mqtt_config"]["BROKER_PORT"],
                self.agent_config["mqtt_config"]["MQTT_USER"],
                self.agent_config["mqtt_config"]["MQTT_PWD"],
                self.agent_config["mqtt_config"]["MQTT_KEEPALIVE"],
                f"FedML_Launch_Master_Agent_@{self.user_name}@_@{self.current_device_id}@_@{str(uuid.uuid4())}@",
                self.topic_last_will,
                json.dumps({"ID": self.edge_id, "status": GeneralConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE})
            )
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
    def _process_job_complete_status(self, run_id, server_id, complete_payload):
        # Complete the job runner
        self._get_job_runner_manager().complete_job_runner(
            run_id, args=self.args, server_id=server_id, request_json=complete_payload,
            run_as_cloud_agent=self.run_as_cloud_agent, run_as_cloud_server=self.run_as_cloud_server,
            use_local_process_as_cloud_server=self.use_local_process_as_cloud_server)

    def generate_agent_instance(self):
        from .master_agent import FedMLLaunchMasterAgent
        return FedMLLaunchMasterAgent()
