
from multiprocessing import Process
from ..comm_utils import sys_utils
from ..comm_utils.job_cleanup import JobCleanup
from ....core.mlops import MLOpsRuntimeLog, MLOpsMetrics
from ..scheduler_core.master_api_daemon import MasterApiDaemon
from ..scheduler_core.account_manager import FedMLAccountManager
from ..scheduler_core.general_constants import GeneralConstants
from abc import ABC, abstractmethod


class FedMLBaseMasterAgent(ABC):

    def __init__(self):
        self.agent_args = None
        self.master_api_daemon = None
        self.master_api_process = None
        self.mlops_metrics = MLOpsMetrics()
        self.status_reporter = None
        self.enable_simulation_cloud_agent = False
        self.use_local_process_as_cloud_server = False
        self.protocol_mgr = None

    def login(
            self, user_id, api_key=None, device_id=None,
            os_name=None, role=None, runner_cmd=None,
            communication_manager=None, sender_message_queue=None,
            status_center_queue=None, sender_message_event=None
    ):
        # Login account
        login_result = FedMLAccountManager.get_instance().login(
            user_id, api_key=api_key, device_id=device_id,
            os_name=os_name, role=role, runner_cmd=runner_cmd
        )
        if login_result is not None:
            self.agent_args = login_result
        else:
            return None

        # Save the bound info
        self._save_agent_info(
            login_result.current_device_id + "." + login_result.os_name, login_result.edge_id)

        # Init the logs for protocol manager
        self._init_logs(login_result, login_result.edge_id)

        # Create the protocol manager to communicate with the slave agents and MLOps.
        self._create_protocol_manager(role, login_result)

        # Initialize the protocol manager
        # noinspection PyBoardException
        try:
            self._initialize_protocol_manager(
                communication_manager=communication_manager,
                sender_message_queue=sender_message_queue,
                status_center_queue=status_center_queue,
                sender_message_event=sender_message_event)
        except Exception as e:
            FedMLAccountManager.write_login_failed_file(is_client=False)
            self.protocol_mgr.stop()
            raise e

        # Start the protocol manager to process the messages from MLOps and slave agents.
        if communication_manager is None:
            self.protocol_mgr.start()

        return login_result

    @staticmethod
    def logout():
        GeneralConstants.cleanup_run_process(None, is_master=True)
        sys_utils.cleanup_all_fedml_server_api_processes()

    def stop(self, kill_process=False):
        if self.protocol_mgr is not None:
            self.protocol_mgr.stop(kill_process=kill_process)

    def _create_protocol_manager(self, role, login_result):
        if self.protocol_mgr is not None:
            return
        self.protocol_mgr = self._generate_protocol_manager_instance(
            login_result, agent_config=login_result.agent_config)
        self.protocol_mgr.run_as_edge_server_and_agent = True \
            if role == FedMLAccountManager.ROLE_EDGE_SERVER else False
        self.protocol_mgr.run_as_cloud_agent = True \
            if role == FedMLAccountManager.ROLE_CLOUD_AGENT or role == FedMLAccountManager.ROLE_GPU_MASTER_SERVER \
            else False
        self.use_local_process_as_cloud_server = True \
            if role == FedMLAccountManager.ROLE_GPU_MASTER_SERVER else self.use_local_process_as_cloud_server
        self.protocol_mgr.run_as_cloud_server = True if role == FedMLAccountManager.ROLE_CLOUD_SERVER else False
        self.protocol_mgr.args = login_result
        self.protocol_mgr.edge_id = login_result.edge_id
        self.protocol_mgr.unique_device_id = login_result.unique_device_id
        self.protocol_mgr.user_name = login_result.user_name
        self.protocol_mgr.agent_config = login_result.agent_config
        self.protocol_mgr.enable_simulation_cloud_agent = self.enable_simulation_cloud_agent
        self.protocol_mgr.use_local_process_as_cloud_server = self.use_local_process_as_cloud_server

    def _initialize_protocol_manager(
            self, communication_manager=None, sender_message_queue=None,
            status_center_queue=None, sender_message_event=None
    ):
        # Init local database
        self._init_database()

        # Initialize the master protocol
        self.protocol_mgr.set_parent_agent(self)
        self.protocol_mgr.initialize(
            communication_manager=communication_manager,
            sender_message_queue=sender_message_queue,
            status_center_queue=status_center_queue,
            sender_message_event=sender_message_event)

        # Report the IDLE status to MLOps
        self.mlops_metrics.report_server_training_status(
            None, GeneralConstants.MSG_MLOPS_SERVER_STATUS_IDLE, edge_id=self.agent_args.edge_id)

        # Cleanup data when startup
        JobCleanup.get_instance().sync_data_on_startup(self.agent_args.edge_id, is_client=False)

        # Start the API server on master agent
        self.master_api_daemon = MasterApiDaemon()
        self.master_api_process = Process(target=self.master_api_daemon.run)
        self.master_api_process.start()

    def _init_logs(self, agent_args, edge_id):
        # Init runtime logs
        in_args = agent_args
        in_args.log_file_dir = self._get_log_file_dir()
        in_args.run_id = 0
        in_args.role = "server"
        in_args.edge_id = edge_id
        in_args.using_mlops = True
        in_args.server_agent_id = edge_id
        MLOpsRuntimeLog.get_instance(in_args).init_logs()

    def get_protocol_manager(self):
        return self.protocol_mgr

    @abstractmethod
    def _get_log_file_dir(self):
        pass

    @abstractmethod
    def _save_agent_info(self, unique_device_id, edge_id):
        pass

    @abstractmethod
    def _init_database(self):
        pass

    @abstractmethod
    def _generate_protocol_manager_instance(self, args, agent_config=None):
        return None

    def start_master_server_instance(self, payload):
        self.protocol_mgr.start_master_server_instance(payload)

    def generate_agent_instance(self):
        return FedMLBaseMasterAgent()

    def process_job_complete_status(self, run_id, topic, payload):
        if self.protocol_mgr is None:
            return
        if topic in self.protocol_mgr.get_subscribed_topics():
            message_handler = self.protocol_mgr.get_listener_handler(topic)
            if message_handler is not None:
                message_handler(topic, payload)
