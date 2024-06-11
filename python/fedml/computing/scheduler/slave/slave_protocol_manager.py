import copy
import os
from ..comm_utils.job_cleanup import JobCleanup
from .base_slave_protocol_manager import FedMLBaseSlaveProtocolManager
from .launch_job_runner_manager import FedMLLaunchJobRunnerManager
from ..model_scheduler.model_device_server import FedMLModelDeviceServerRunner
from ..model_scheduler.model_device_client import FedMLModelDeviceClientRunner


class FedMLLaunchSlaveProtocolManager(FedMLBaseSlaveProtocolManager):

    def __init__(self, args, agent_config=None):
        FedMLBaseSlaveProtocolManager.__init__(self, args, agent_config=agent_config)

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

        # Get the environment variables
        infer_host = os.getenv("FEDML_INFER_HOST", None)
        infer_redis_addr = os.getenv("FEDML_INFER_REDIS_ADDR", None)
        infer_redis_port = os.getenv("FEDML_INFER_REDIS_PORT", None)
        infer_redis_password = os.getenv("FEDML_INFER_REDIS_PASSWORD", None)
        model_client_num = os.getenv("FEDML_MODEL_WORKER_NUM", None)

        # Start deploy master agent and slave agent
        in_args = copy.deepcopy(self.args)
        if self.model_device_client_edge_id_list is None:
            self.model_device_client_edge_id_list = list()
        if self.model_device_client_list is None:
            model_client_num = 1 if model_client_num is None else int(model_client_num)
            self.model_device_client_list = list()
            for client_index in range(model_client_num):
                model_device_client = FedMLModelDeviceClientRunner(
                    in_args, f"{in_args.current_device_id}_{client_index + 1}", in_args.os_name,
                    in_args.is_from_docker, self.agent_config)
                if infer_host is not None:
                    model_device_client.infer_host = infer_host
                if infer_redis_addr is not None:
                    model_device_client.redis_addr = infer_redis_addr
                if infer_redis_port is not None:
                    model_device_client.redis_port = infer_redis_port
                if infer_redis_password is not None:
                    model_device_client.redis_password = infer_redis_password
                self.model_device_client_list.append(model_device_client)
                self.model_device_client_edge_id_list.append(model_device_client.bind_device())

        self.args = copy.deepcopy(in_args)
        if self.model_device_server is None:
            self.model_device_server = FedMLModelDeviceServerRunner(in_args, in_args.current_device_id,
                                                                    in_args.os_name, in_args.is_from_docker,
                                                                    self.agent_config)
            if infer_host is not None:
                self.model_device_server.infer_host = infer_host
            if infer_redis_addr is not None:
                self.model_device_server.redis_addr = infer_redis_addr
            if infer_redis_port is not None:
                self.model_device_server.redis_port = infer_redis_port
            if infer_redis_password is not None:
                self.model_device_server.redis_password = infer_redis_password

            self.model_device_server_id = self.model_device_server.bind_device()

        # Save the deployed master and worker id list to the environment variable.
        os.environ["FEDML_DEPLOY_MASTER_ID"] = str(self.model_device_server_id)
        os.environ["FEDML_DEPLOY_WORKER_IDS"] = str(self.model_device_client_edge_id_list)

        # Start the monitor process
        self.args = copy.deepcopy(in_args)
        self.mlops_metrics.stop_device_realtime_perf()
        self.mlops_metrics.report_device_realtime_perf(self.args, self.args.agent_config["mqtt_config"])
        pass
