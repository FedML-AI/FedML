
import json
import os
from ..comm_utils import sys_utils
from ..comm_utils.run_process_utils import RunProcessUtils
from ..comm_utils.sys_utils import get_python_program
from ....core.mlops import MLOpsRuntimeLog, MLOpsMetrics
from .client_data_interface import ClientConstants
from ..scheduler_core.account_manager import FedMLAccountManager
from ..scheduler_core.general_constants import GeneralConstants
from abc import ABC, abstractmethod


class FedMLBaseSlaveAgent(ABC):
    CLIENT_API_CMD = "fedml.computing.scheduler.slave.client_api:api"

    def __init__(self):
        self.agent_args = None
        self.local_api_process = None
        self.process = None
        self.cur_dir = os.path.split(os.path.realpath(__file__))[0]
        self.mlops_metrics = MLOpsMetrics()
        self.protocol_mgr = None

    def login(
            self, userid, api_key=None, device_id=None,
            os_name=None, need_to_check_gpu=False, role=None,
            communication_manager=None, sender_message_queue=None,
            status_center_queue=None, sender_message_event=None
    ):
        # Preprocess the login args
        if need_to_check_gpu:
            gpu_count, _ = sys_utils.get_gpu_count_vendor()
            if gpu_count <= 0:
                print("We can't find any gpu device on your machine. \n"
                      "With the gpu_supplier(-g) option, you need to check if your machine "
                      "has nvidia GPUs and installs CUDA related drivers.")
                return None

        # Login account
        login_result = FedMLAccountManager.get_instance().login(
            userid, api_key=api_key, device_id=device_id,
            os_name=os_name, role=role
        )
        if login_result is not None:
            self.agent_args = login_result
        else:
            return None

        # Save the bound info
        self._save_agent_info(login_result.current_device_id + "." + login_result.os_name, login_result.edge_id)

        # Init the logs for protocol manager
        self._init_logs(login_result, login_result.edge_id)

        # Create the protocol manager to communicate with the slave agents and MLOps.
        self._create_protocol_manager(login_result)

        # Initialize the protocol manager
        # noinspection PyBoardException
        try:
            self._initialize_protocol_manager(
                communication_manager=communication_manager,
                sender_message_queue=sender_message_queue,
                status_center_queue=status_center_queue,
                sender_message_event=sender_message_event)
        except Exception as e:
            FedMLAccountManager.write_login_failed_file(is_client=True)
            self.protocol_mgr.stop()
            raise e

        return login_result

    def start(self):
        # Start the protocol manager to process the messages from MLOps and slave agents.
        self.protocol_mgr.start()

    @staticmethod
    def logout():
        GeneralConstants.cleanup_run_process(None)
        sys_utils.cleanup_all_fedml_client_api_processes()

    def _create_protocol_manager(self, login_result):
        if self.protocol_mgr is not None:
            return
        self.protocol_mgr = self._generate_protocol_manager_instance(
            login_result, agent_config=login_result.agent_config)
        self.protocol_mgr.args = login_result
        self.protocol_mgr.edge_id = login_result.edge_id
        self.protocol_mgr.unique_device_id = login_result.unique_device_id
        self.protocol_mgr.user_name = login_result.user_name
        self.protocol_mgr.agent_config = login_result.agent_config

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

        # Start the client API process
        self._start_slave_api()

    def _init_logs(self, login_result, edge_id):
        # Init runtime logs
        in_args = login_result
        in_args.log_file_dir = self._get_log_file_dir()
        in_args.run_id = 0
        in_args.role = "client"
        client_ids = list()
        client_ids.append(edge_id)
        in_args.client_id_list = json.dumps(client_ids)
        in_args.using_mlops = True
        MLOpsRuntimeLog.get_instance(in_args).init_logs()

    def _start_slave_api(self):
        # Start the local API services
        client_api_cmd = FedMLBaseSlaveAgent.CLIENT_API_CMD
        client_api_pids = RunProcessUtils.get_pid_from_cmd_line(client_api_cmd)
        if client_api_pids is None or len(client_api_pids) <= 0:
            python_program = get_python_program()
            cur_dir = os.path.dirname(__file__)
            fedml_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
            self.local_api_process = ClientConstants.exec_console_with_script(
                "{} -m uvicorn {} --host 0.0.0.0 --port {} "
                "--reload --reload-delay 3 --reload-dir {} --log-level critical".format(
                    python_program, client_api_cmd, ClientConstants.LOCAL_CLIENT_API_PORT, fedml_base_dir),
                should_capture_stdout=False,
                should_capture_stderr=False
            )

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

    def save_deploy_ids(self, deploy_master_edge_id=None, deploy_slave_edge_id=None):
        self.protocol_mgr.save_deploy_ids(
            deploy_master_edge_id=deploy_master_edge_id, deploy_slave_edge_id=deploy_slave_edge_id)

