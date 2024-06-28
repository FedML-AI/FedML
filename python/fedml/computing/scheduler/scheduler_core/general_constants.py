import logging
import os

from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.comm_utils.run_process_utils import RunProcessUtils
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.model_scheduler import device_client_constants
from fedml.computing.scheduler.model_scheduler import device_server_constants


class GeneralConstants:
    MSG_TOPIC_REQUEST_JOB_STATUS_PREFIX = f"anywhere/master_agent/request_job_status/"
    MSG_TOPIC_REPORT_DEVICE_STATUS_IN_JOB = f"slave_job/slave_agent/report_device_status_in_job"
    MSG_TOPIC_SEND_TRAINING_REQUEST_TO_EDGES = "job_runner/master_protocol_manager/send_training_request_to_edges"

    CLIENT_SHELL_BASH = SchedulerConstants.CLIENT_SHELL_BASH
    CLIENT_SHELL_PS = SchedulerConstants.CLIENT_SHELL_PS
    PLATFORM_WINDOWS = "Windows"

    MSG_MLOPS_CLIENT_STATUS_OFFLINE = "OFFLINE"
    MSG_MLOPS_CLIENT_STATUS_PROVISIONING = "PROVISIONING"
    MSG_MLOPS_CLIENT_STATUS_IDLE = "IDLE"
    MSG_MLOPS_CLIENT_STATUS_UPGRADING = "UPGRADING"
    MSG_MLOPS_CLIENT_STATUS_QUEUED = "QUEUED"
    MSG_MLOPS_CLIENT_STATUS_INITIALIZING = "INITIALIZING"
    MSG_MLOPS_CLIENT_STATUS_TRAINING = "TRAINING"
    MSG_MLOPS_CLIENT_STATUS_RUNNING = "RUNNING"
    MSG_MLOPS_CLIENT_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_CLIENT_STATUS_KILLED = "KILLED"
    MSG_MLOPS_CLIENT_STATUS_FAILED = "FAILED"
    MSG_MLOPS_CLIENT_STATUS_EXCEPTION = "EXCEPTION"
    MSG_MLOPS_CLIENT_STATUS_FINISHED = "FINISHED"

    MSG_MLOPS_SERVER_STATUS_OFFLINE = "OFFLINE"
    MSG_MLOPS_SERVER_STATUS_PROVISIONING = "PROVISIONING"
    MSG_MLOPS_SERVER_STATUS_IDLE = "IDLE"
    MSG_MLOPS_SERVER_STATUS_UPGRADING = "UPGRADING"
    MSG_MLOPS_SERVER_STATUS_STARTING = "STARTING"
    MSG_MLOPS_SERVER_STATUS_RUNNING = "RUNNING"
    MSG_MLOPS_SERVER_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_SERVER_STATUS_KILLED = "KILLED"
    MSG_MLOPS_SERVER_STATUS_FAILED = "FAILED"
    MSG_MLOPS_SERVER_STATUS_FINISHED = "FINISHED"
    MSG_MLOPS_SERVER_STATUS_EXCEPTION = "EXCEPTION"

    MSG_MODELOPS_DEPLOYMENT_STATUS_INITIALIZING = "INITIALIZING"
    MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYING = "DEPLOYING"
    MSG_MODELOPS_DEPLOYMENT_STATUS_INFERRING = "INFERRING"
    MSG_MODELOPS_DEPLOYMENT_STATUS_OVERLOAD = "OVERLOAD"
    MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED = "FAILED"
    MSG_MODELOPS_DEPLOYMENT_STATUS_RESCALING = "RESCALING"
    MSG_MODELOPS_DEPLOYMENT_STATUS_UPDATING = "UPDATING"
    MSG_MODELOPS_DEPLOYMENT_STATUS_UPDATING_FAILED = "UPDATING_FAILED"
    MSG_MODELOPS_DEPLOYMENT_STATUS_ABORTING = "ABORTING"
    MSG_MODELOPS_DEPLOYMENT_STATUS_ABORTED = "ABORTED"
    MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED = "DEPLOYED"
    MSG_MODELOPS_DEPLOYMENT_STATUS_KILLED = "KILLED"

    MASTER_LOGIN_PROGRAM = "server_login.py"
    SLAVE_LOGIN_PROGRAM = "client_login.py"

    CONFIG_KEY_AUTO_DETECT_PUBLIC_IP = "auto_detect_public_ip"
    FEDML_OTA_CMD_UPGRADE = "upgrade"
    FEDML_OTA_CMD_RESTART = "restart"

    FEDML_LOG_SOURCE_TYPE_MODEL_END_POINT = "MODEL_END_POINT"
    FEDML_PROCESS_NAME_PREFIX = "fedml-process-"
    FEDML_LAUNCH_MASTER_JOB_RUNNER_TAG = "launch-master-job-runner"
    FEDML_LAUNCH_SLAVE_JOB_RUNNER_TAG = "launch-slave-job-runner"
    FEDML_LAUNCH_MASTER_USER_JOB_TAG = "launch-master-user-job"
    FEDML_DEPLOY_MASTER_JOB_RUNNER_TAG = "deploy-master-job-runner"
    FEDML_DEPLOY_SLAVE_JOB_RUNNER_TAG = "deploy-slave-job-runner"
    FEDML_DEPLOY_MASTER_USER_JOB_TAG = "deploy-master-user-job"
    FEDML_MESSAGE_CENTER_LISTENER_TAG = "message-center-listener"
    FEDML_MESSAGE_CENTER_SENDER_TAG = "message-center-sender"
    FEDML_STATUS_CENTER_TAG = "status-center"
    FEDML_LOG_PROCESS_TAG = "log"
    FEDML_MONITOR_PROCESS_TAG = "monitor"

    FEDML_TOPIC_STATUS_CENTER_STOP = "anywhere/status_center/stop"

    @staticmethod
    def get_package_unzip_dir(package_download_dir):
        package_unzip_dir = package_download_dir
        if not os.path.exists(package_unzip_dir):
            os.makedirs(package_unzip_dir, exist_ok=True)
        return package_unzip_dir

    @staticmethod
    def get_filename_and_extension(url):
        return ClientConstants.get_filename_and_extension(url)

    @staticmethod
    def generate_yaml_doc(run_config_object, yaml_file):
        ClientConstants.generate_yaml_doc(run_config_object, yaml_file)

    @staticmethod
    def execute_commands_with_live_logs(cmds, join='&&', should_write_log_file=True,
                                        callback=None, error_processor=None):
        return ClientConstants.execute_commands_with_live_logs(
            cmds, join=join, should_write_log_file=should_write_log_file,
            callback=callback, error_processor=error_processor
        )

    @staticmethod
    def cleanup_run_process(run_id, is_master=False):
        if is_master:
            ServerConstants.cleanup_run_process(run_id)
        else:
            ClientConstants.cleanup_run_process(run_id)

    @staticmethod
    def cleanup_learning_process(run_id, data_dir=None):
        RunProcessUtils.cleanup_run_process(
            run_id, data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix=SchedulerConstants.RUN_PROCESS_TYPE_USER_PROCESS)

    @staticmethod
    def cleanup_bootstrap_process(run_id, data_dir=None):
        RunProcessUtils.cleanup_run_process(
            run_id, data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix=SchedulerConstants.RUN_PROCESS_TYPE_BOOTSTRAP_PROCESS)

    @staticmethod
    def save_learning_process(run_id, learning_id, data_dir=None):
        RunProcessUtils.save_run_process(
            run_id, learning_id, data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix=SchedulerConstants.RUN_PROCESS_TYPE_USER_PROCESS)

    @staticmethod
    def save_bootstrap_process(run_id, process_id, data_dir=None):
        RunProcessUtils.save_run_process(
            run_id, process_id, data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix=SchedulerConstants.RUN_PROCESS_TYPE_BOOTSTRAP_PROCESS)

    @staticmethod
    def save_run_process(run_id, process_id, is_master=False):
        RunProcessUtils.save_run_process(
            run_id, process_id, ServerConstants.get_data_dir() if is_master else ClientConstants.get_data_dir(),
            ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)

    @staticmethod
    def get_learning_process_list(run_id, is_master=False):
        return RunProcessUtils.get_run_process_list(
            run_id, ServerConstants.get_data_dir() if is_master else ClientConstants.get_data_dir(),
            ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix=SchedulerConstants.RUN_PROCESS_TYPE_USER_PROCESS)

    @staticmethod
    def get_launch_fedml_home_dir(is_master=False):
        return ServerConstants.get_fedml_home_dir() if is_master else ClientConstants.get_fedml_home_dir()

    @staticmethod
    def get_deploy_fedml_home_dir(is_master=False):
        return device_server_constants.ServerConstants.get_fedml_home_dir() if is_master \
            else device_client_constants.ClientConstants.get_fedml_home_dir()

    @staticmethod
    def get_launch_log_file_dir(is_master=False):
        return ServerConstants.get_log_file_dir() if is_master else ClientConstants.get_log_file_dir()

    @staticmethod
    def get_deploy_log_file_dir(is_master=False):
        return device_server_constants.ServerConstants.get_log_file_dir() if is_master \
            else device_client_constants.ClientConstants.get_log_file_dir()

    @staticmethod
    def get_launch_data_dir(is_master=False):
        return ServerConstants.get_data_dir() if is_master else ClientConstants.get_data_dir()

    @staticmethod
    def get_deploy_data_dir(is_master=False):
        return device_server_constants.ServerConstants.get_data_dir() if is_master \
            else device_client_constants.ClientConstants.get_data_dir()

    @staticmethod
    def get_deploy_docker_location_file(is_master=False):
        return device_server_constants.ServerConstants.get_docker_location_file() if is_master \
            else device_client_constants.ClientConstants.get_docker_location_file()

    @staticmethod
    def get_launch_docker_location_file(is_master=False):
        return ServerConstants.get_docker_location_file() if is_master \
            else ClientConstants.get_docker_location_file()

    @staticmethod
    def get_local_ip():
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        conn = s.connect(('8.8.8.8', 53))
        ip = s.getsockname()[0]
        s.close()
        return ip

    @staticmethod
    def get_public_ip():
        import requests
        ip = None
        try:
            ip = requests.get('https://checkip.amazonaws.com').text.strip()
        except Exception as e:
            logging.info("Failed to get public ip: {}".format(e))
        return ip

    @staticmethod
    def get_ip_address(request_json, infer_host=None):
        # OPTION 1: Use local ip
        # ip = GeneralConstants.get_local_ip()
        #
        # # OPTION 2: Auto detect public ip
        # if "parameters" in request_json and \
        #         GeneralConstants.CONFIG_KEY_AUTO_DETECT_PUBLIC_IP in request_json["parameters"] and \
        #         request_json["parameters"][GeneralConstants.CONFIG_KEY_AUTO_DETECT_PUBLIC_IP]:
        ip = GeneralConstants.get_public_ip()
        logging.info("Auto detect public ip for master: " + ip)

        # OPTION 3: Use user indicated ip
        if infer_host is not None and infer_host != "127.0.0.1" and infer_host != "localhost":
            ip = infer_host

        return ip

    @staticmethod
    def get_topic_complete_job(server_id):
        topic_complete_job = f"status_center/master_agent_{server_id}/complete_job"
        return topic_complete_job

    @staticmethod
    def get_payload_complete_job(run_id, server_id):
        payload_complete_job = {"runId": run_id, "serverId": server_id}
        return payload_complete_job

    @staticmethod
    def get_process_name(process_tag, run_id=None, edge_id=None):
        return f'{GeneralConstants.FEDML_PROCESS_NAME_PREFIX}{process_tag}'\
               f'{"-run-" + str(run_id) if run_id is not None and int(run_id) != 0 else ""}'\
               f'{"-edge-" + str(edge_id) if edge_id is not None else ""}'

    @staticmethod
    def get_process_name_with_prefix(process_prefix, run_id=None, edge_id=None):
        return f"{process_prefix}-run-{run_id}-edge-{edge_id}"

    @staticmethod
    def get_launch_master_job_process_name(run_id, edge_id):
        return GeneralConstants.get_process_name(
            GeneralConstants.FEDML_LAUNCH_MASTER_JOB_RUNNER_TAG, run_id, edge_id)

    @staticmethod
    def get_launch_slave_job_process_name(run_id, edge_id):
        return GeneralConstants.get_process_name(
            GeneralConstants.FEDML_LAUNCH_SLAVE_JOB_RUNNER_TAG, run_id, edge_id)

    @staticmethod
    def get_launch_master_user_process_name(run_id, edge_id):
        return GeneralConstants.get_process_name(
            GeneralConstants.FEDML_LAUNCH_MASTER_USER_JOB_TAG, run_id, edge_id)

    @staticmethod
    def get_deploy_master_job_process_name(run_id, edge_id):
        return GeneralConstants.get_process_name(
            GeneralConstants.FEDML_DEPLOY_MASTER_JOB_RUNNER_TAG, run_id, edge_id)

    @staticmethod
    def get_deploy_slave_job_process_name(run_id, edge_id):
        return GeneralConstants.get_process_name(
            GeneralConstants.FEDML_DEPLOY_SLAVE_JOB_RUNNER_TAG, run_id, edge_id)

    @staticmethod
    def get_deploy_master_user_process_name(run_id, edge_id):
        return GeneralConstants.get_process_name(
            GeneralConstants.FEDML_DEPLOY_MASTER_USER_JOB_TAG, run_id, edge_id)

    @staticmethod
    def get_log_process_name(run_id, edge_id):
        return GeneralConstants.get_process_name(
            GeneralConstants.FEDML_LOG_PROCESS_TAG, run_id, edge_id)

    @staticmethod
    def get_message_center_listener_process_name(message_center_name):
        return f"{GeneralConstants.FEDML_PROCESS_NAME_PREFIX}{GeneralConstants.FEDML_MESSAGE_CENTER_LISTENER_TAG}-{message_center_name}"

    @staticmethod
    def get_message_center_sender_process_name(message_center_name):
        return f"{GeneralConstants.FEDML_PROCESS_NAME_PREFIX}{GeneralConstants.FEDML_MESSAGE_CENTER_SENDER_TAG}-{message_center_name}"

    @staticmethod
    def get_status_center_process_name(status_center_tag):
        return f"{GeneralConstants.FEDML_PROCESS_NAME_PREFIX}{GeneralConstants.FEDML_STATUS_CENTER_TAG}-{status_center_tag}"

    @staticmethod
    def get_monitor_process_name(monitor_tag, run_id, edge_id):
        return GeneralConstants.get_process_name(
            f"{GeneralConstants.FEDML_MONITOR_PROCESS_TAG}-{monitor_tag}", run_id, edge_id)
