
import os
import platform
import signal
import subprocess
import sys
from os.path import expanduser

import psutil
import yaml

from ..comm_utils.constants import SchedulerConstants
from ..comm_utils.run_process_utils import RunProcessUtils
from ..comm_utils.yaml_utils import load_yaml_config

import fedml

class ServerConstants(object):
    MSG_MLOPS_SERVER_STATUS_OFFLINE = "OFFLINE"
    MSG_MLOPS_SERVER_STATUS_IDLE = "IDLE"
    MSG_MLOPS_SERVER_STATUS_UPGRADING = "UPGRADING"
    MSG_MLOPS_SERVER_STATUS_STARTING = "STARTING"
    MSG_MLOPS_SERVER_STATUS_RUNNING = "RUNNING"
    MSG_MLOPS_SERVER_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_SERVER_STATUS_KILLED = "KILLED"
    MSG_MLOPS_SERVER_STATUS_FAILED = "FAILED"
    MSG_MLOPS_SERVER_STATUS_FINISHED = "FINISHED"
    MSG_MLOPS_SERVER_STATUS_EXCEPTION = "EXCEPTION"

    # Device Status
    MSG_MLOPS_DEVICE_STATUS_IDLE = "IDLE"
    MSG_MLOPS_DEVICE_STATUS_UPGRADING = "UPGRADING"
    MSG_MLOPS_DEVICE_STATUS_RUNNING = "RUNNING"
    MSG_MLOPS_DEVICE_STATUS_OFFLINE = "OFFLINE"

    # Run Status
    MSG_MLOPS_RUN_STATUS_QUEUED = "QUEUED"
    MSG_MLOPS_RUN_STATUS_STARTING = "STARTING"
    MSG_MLOPS_RUN_STATUS_RUNNING = "RUNNING"
    MSG_MLOPS_RUN_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_RUN_STATUS_KILLED = "KILLED"
    MSG_MLOPS_RUN_STATUS_FAILED = "FAILED"
    MSG_MLOPS_RUN_STATUS_FINISHED = "FINISHED"

    LOCAL_HOME_RUNNER_DIR_NAME = 'fedml-model-server'
    LOCAL_RUNNER_INFO_DIR_NAME = 'runner_infos'
    LOCAL_PACKAGE_HOME_DIR_NAME = "fedml_packages"

    FEDML_OTA_CMD_UPGRADE = "upgrade"
    FEDML_OTA_CMD_RESTART = "restart"

    LOCAL_SERVER_API_PORT = 40806

    # Constants for models
    K8S_DEPLOYMENT_MASTER_HOST_HOME_DIR = "/home/fedml-server"
    K8S_DEPLOYMENT_SLAVE_HOST_HOME_DIR = "/home/fedml-client"
    K8S_DEPLOYMENT_MASTER_MOUNT_HOME_DIR = "/home/fedml/fedml-server"
    K8S_DEPLOYMENT_SLAVE_MOUNT_HOME_DIR = "/home/fedml/fedml-client"

    INFERENCE_HTTP_PORT = 8000
    INFERENCE_GRPC_PORT = 8001
    INFERENCE_METRIC_PORT = 8002

    SERVER_LOGIN_PROGRAM = "device_server_login.py"

    FEDML_LOG_SOURCE_TYPE_MODEL_END_POINT = "MODEL_END_POINT"

    INFERENCE_CONVERTOR_IMAGE = "public.ecr.aws/x6k8q1x9/fedml-inference-converter:latest"
    INFERENCE_SERVER_IMAGE = "public.ecr.aws/x6k8q1x9/fedml-inference-backend:latest"

    INFERENCE_SERVER_STARTED_TAG = "Started HTTPService at 0.0.0.0:"
    INFERENCE_ENGINE_TYPE_ONNX = "onnx"
    INFERENCE_ENGINE_TYPE_TENSORRT = "tensorrt"
    INFERENCE_MODEL_VERSION = "1"
    INFERENCE_INFERENCE_SERVER_VERSION = "v2"

    MSG_MODELOPS_DEPLOYMENT_STATUS_INITIALIZING = "INITIALIZING"
    MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYING = "DEPLOYING"
    MSG_MODELOPS_DEPLOYMENT_STATUS_INFERRING = "INFERRING"
    MSG_MODELOPS_DEPLOYMENT_STATUS_OVERLOAD = "OVERLOAD"
    MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED = "FAILED"
    MSG_MODELOPS_DEPLOYMENT_STATUS_RESCALING = "RESCALING"
    MSG_MODELOPS_DEPLOYMENT_STATUS_UPDATING = "UPDATING"
    MSG_MODELOPS_DEPLOYMENT_STATUS_ROLLBACK = "ROLLBACK"
    MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED = "DEPLOYED"

    MODEL_REQUIRED_MODEL_CONFIG_FILE = "fedml_model_config.yaml"
    MODEL_REQUIRED_MODEL_BIN_FILE = "fedml_model.bin"
    MODEL_REQUIRED_MODEL_README_FILE = "README.md"

    CMD_TYPE_CONVERT_MODEL = "convert_model"
    CMD_TYPE_RUN_TRITON_SERVER = "run_triton_server"
    FEDML_CONVERT_MODEL_CONTAINER_NAME_PREFIX = "fedml_convert_model_container"
    FEDML_TRITON_SERVER_CONTAINER_NAME_PREFIX = "fedml_triton_server_container"
    FEDML_CONVERTED_MODEL_DIR_NAME = "triton_models"
    FEDML_MODEL_SERVING_REPO_SCAN_INTERVAL = 3

    FEDML_RUNNING_SOURCE_ENV_NAME = "FEDML_RUNNING_SOURCE"
    FEDML_RUNNING_SOURCE_ENV_VALUE_K8S = "k8s"

    AUTO_DETECT_PUBLIC_IP = "auto_detect_public_ip"
    MODEL_INFERENCE_DEFAULT_PORT = 5001
    # -----End-----

    MODEL_DEPLOYMENT_STAGE1 = {"index": 1, "text": "ReceivedRequest"}
    MODEL_DEPLOYMENT_STAGE2 = {"index": 2, "text": "Initializing"}
    MODEL_DEPLOYMENT_STAGE3 = {"index": 3, "text": "StartRunner"}
    MODEL_DEPLOYMENT_STAGE4 = {"index": 4, "text": "ForwardRequestToSlave"}
    MODEL_DEPLOYMENT_STAGE5 = {"index": 5, "text": "StartInferenceIngress"}

    LOGIN_MODE_ON_PREMISE_MASTER_INDEX = 0
    LOGIN_MODE_FEDML_CLOUD_MASTER_INDEX = 1
    LOGIN_MODE_PUBLIC_CLOUD_MASTER_INDEX = 2
    LOGIN_MODE_INFERENCE_INSTANCE_INDEX = 3
    login_role_list = ["md.on_premise_device.master", "md.fedml_cloud_device.master", "md.pubic_cloud_device.master",
                       "md.inference_instance_device"]

    MODEL_DATA_TYPE_INT = "int"
    MODEL_DATA_TYPE_FLOAT = "float"
    MODEL_DATA_TYPE_STR = "str"
    MODEL_DATA_TYPE_MAPPING = {"TYPE_BOOL": MODEL_DATA_TYPE_INT, "TYPE_UINT8": MODEL_DATA_TYPE_INT,
                               "TYPE_UINT16": MODEL_DATA_TYPE_INT, "TYPE_UINT32": MODEL_DATA_TYPE_INT,
                               "TYPE_UINT64": MODEL_DATA_TYPE_INT, "TYPE_INT8": MODEL_DATA_TYPE_INT,
                               "TYPE_INT16": MODEL_DATA_TYPE_INT, "TYPE_INT32": MODEL_DATA_TYPE_INT,
                               "TYPE_INT64": MODEL_DATA_TYPE_INT, "TYPE_FP16": MODEL_DATA_TYPE_FLOAT,
                               "TYPE_FP32": MODEL_DATA_TYPE_FLOAT, "TYPE_FP64": MODEL_DATA_TYPE_FLOAT,
                               "TYPE_STRING": MODEL_DATA_TYPE_STR, "TYPE_BF16": MODEL_DATA_TYPE_INT,
                               "BOOL": MODEL_DATA_TYPE_INT, "UINT8": MODEL_DATA_TYPE_INT,
                               "UINT16": MODEL_DATA_TYPE_INT, "UINT32": MODEL_DATA_TYPE_INT,
                               "UINT64": MODEL_DATA_TYPE_INT, "INT8": MODEL_DATA_TYPE_INT,
                               "INT16": MODEL_DATA_TYPE_INT, "INT32": MODEL_DATA_TYPE_INT,
                               "INT64": MODEL_DATA_TYPE_INT, "FP16": MODEL_DATA_TYPE_FLOAT,
                               "FP32": MODEL_DATA_TYPE_FLOAT, "FP64": MODEL_DATA_TYPE_FLOAT,
                               "STRING": MODEL_DATA_TYPE_STR, "BF16": MODEL_DATA_TYPE_INT}

    @staticmethod
    def get_fedml_home_dir():
        home_dir = expanduser("~")
        fedml_home_dir = os.path.join(home_dir, ".fedml", ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME)
        if not os.path.exists(fedml_home_dir):
            os.makedirs(fedml_home_dir, exist_ok=True)
        return fedml_home_dir

    @staticmethod
    def get_log_file_dir():
        log_file_dir = os.path.join(ServerConstants.get_fedml_home_dir(), "fedml", "logs")
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir, exist_ok=True)
        return log_file_dir

    @staticmethod
    def get_data_dir():
        data_dir = os.path.join(ServerConstants.get_fedml_home_dir(), "fedml", "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        return data_dir

    @staticmethod
    def get_package_download_dir():
        package_download_dir = os.path.join(ServerConstants.get_fedml_home_dir(),
                                            ServerConstants.LOCAL_PACKAGE_HOME_DIR_NAME)
        if not os.path.exists(package_download_dir):
            os.makedirs(package_download_dir, exist_ok=True)
        return package_download_dir

    @staticmethod
    def get_package_unzip_dir(run_id, package_url):
        package_unzip_dir_name = "run_{}_{}".format(str(run_id),
                                                    str(os.path.basename(package_url)).split('.')[0])
        package_unzip_dir = os.path.join(ServerConstants.get_package_download_dir(),
                                         package_unzip_dir_name)
        if not os.path.exists(package_unzip_dir):
            os.makedirs(package_unzip_dir, exist_ok=True)
        return package_unzip_dir

    @staticmethod
    def get_package_run_dir(run_id, package_url, package_name):
        package_file_no_extension = str(package_name).split('.')[0]
        package_run_dir = os.path.join(ServerConstants.get_package_unzip_dir(run_id, package_url),
                                       package_file_no_extension)
        if not os.path.exists(package_run_dir):
            os.makedirs(package_run_dir, exist_ok=True)
        return package_run_dir

    @staticmethod
    def get_database_dir():
        database_dir = os.path.join(ServerConstants.get_data_dir(), "database")
        if not os.path.exists(database_dir):
            os.makedirs(database_dir, exist_ok=True)
        return database_dir

    @staticmethod
    def get_model_dir():
        model_file_dir = os.path.join(ServerConstants.get_fedml_home_dir(), "fedml", "models")
        if not os.path.exists(model_file_dir):
            os.makedirs(model_file_dir, exist_ok=True)
        return model_file_dir

    @staticmethod
    def get_model_package_dir():
        model_packages_dir = os.path.join(ServerConstants.get_fedml_home_dir(), "fedml", "model_packages")
        if not os.path.exists(model_packages_dir):
            os.makedirs(model_packages_dir, exist_ok=True)
        return model_packages_dir

    @staticmethod
    def get_k8s_master_host_dir(current_dir):
        if not ServerConstants.is_running_on_k8s():
            return current_dir

        if str(current_dir).startswith(ServerConstants.K8S_DEPLOYMENT_MASTER_MOUNT_HOME_DIR):
            return str(current_dir).replace(ServerConstants.K8S_DEPLOYMENT_MASTER_MOUNT_HOME_DIR,
                                            ServerConstants.K8S_DEPLOYMENT_MASTER_HOST_HOME_DIR)
        return current_dir

    @staticmethod
    def get_k8s_slave_host_dir(current_dir):
        if not ServerConstants.is_running_on_k8s():
            return current_dir

        if str(current_dir).startswith(ServerConstants.K8S_DEPLOYMENT_SLAVE_MOUNT_HOME_DIR):
            return str(current_dir).replace(ServerConstants.K8S_DEPLOYMENT_SLAVE_MOUNT_HOME_DIR,
                                            ServerConstants.K8S_DEPLOYMENT_SLAVE_HOST_HOME_DIR)
        return current_dir

    @staticmethod
    def is_running_on_k8s():
        running_source = os.getenv(ServerConstants.FEDML_RUNNING_SOURCE_ENV_NAME, default=None)
        if running_source is not None and running_source == ServerConstants.FEDML_RUNNING_SOURCE_ENV_VALUE_K8S:
            return True
        return False

    @staticmethod
    def get_model_serving_dir():
        model_file_dir = os.path.join(ServerConstants.get_fedml_home_dir(), "fedml", "models_serving")
        if not os.path.exists(model_file_dir):
            os.makedirs(model_file_dir, exist_ok=True)
        return model_file_dir

    @staticmethod
    def get_model_ops_list_url(config_version="release"):
        model_ops_url = "{}/api/v1/model/listFromCli".format(ServerConstants.get_model_ops_url(config_version))
        return model_ops_url

    @staticmethod
    def get_model_ops_upload_url(config_version="release"):
        model_ops_url = "{}/api/v1/model/createFromCli".format(ServerConstants.get_model_ops_url(config_version))
        return model_ops_url

    @staticmethod
    def get_model_ops_url(config_version="release"):
        url = fedml._get_backend_service()
        return f"{url}/fedmlModelServer"

    @staticmethod
    def get_model_ops_deployment_url(config_version="release"):
        model_ops_url = "{}/api/v1/endpoint/createFromCli".format(ServerConstants.get_model_ops_url(config_version))
        return model_ops_url

    @staticmethod
    def get_running_model_name(end_point_name, model_name, model_version, end_point_id=None, model_id=None):
        running_model_name = "model_endpoint_{}_model_{}_ver_-{}".format(end_point_name, model_name, model_version)
        running_model_name = running_model_name.replace(' ', '-')
        running_model_name = running_model_name.replace(':', '-')
        return running_model_name

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
            print("Failed to get public ip: {}".format(e))
        return ip

    @staticmethod
    def cleanup_run_process(run_id):
        RunProcessUtils.cleanup_run_process(
            run_id, ServerConstants.get_data_dir(), ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)

    @staticmethod
    def save_run_process(run_id, process_id):
        RunProcessUtils.save_run_process(
            run_id, process_id, ServerConstants.get_data_dir(), ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)

    @staticmethod
    def cleanup_learning_process(run_id):
        RunProcessUtils.cleanup_run_process(
            run_id, ServerConstants.get_data_dir(), ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix=SchedulerConstants.RUN_PROCESS_TYPE_USER_PROCESS)

    @staticmethod
    def save_learning_process(run_id, learning_id):
        RunProcessUtils.save_run_process(
            run_id, learning_id, ServerConstants.get_data_dir(), ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix=SchedulerConstants.RUN_PROCESS_TYPE_USER_PROCESS)

    @staticmethod
    def cleanup_bootstrap_process(run_id):
        RunProcessUtils.cleanup_run_process(
            run_id, ServerConstants.get_data_dir(), ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix=SchedulerConstants.RUN_PROCESS_TYPE_BOOTSTRAP_PROCESS)

    @staticmethod
    def save_bootstrap_process(run_id, process_id):
        RunProcessUtils.save_run_process(
            run_id, process_id, ServerConstants.get_data_dir(), ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix=SchedulerConstants.RUN_PROCESS_TYPE_BOOTSTRAP_PROCESS)

    @staticmethod
    def save_runner_infos(unique_device_id, edge_id, run_id=None):
        home_dir = expanduser("~")
        local_pkg_data_dir = ServerConstants.get_data_dir()
        os.makedirs(local_pkg_data_dir, exist_ok=True)
        os.makedirs(os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME), exist_ok=True)

        runner_info_file = os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                        "runner_infos.yaml")
        running_info = dict()
        running_info["unique_device_id"] = str(unique_device_id)
        running_info["edge_id"] = str(edge_id)
        running_info["run_id"] = run_id
        ServerConstants.generate_yaml_doc(running_info, runner_info_file)

    @staticmethod
    def get_docker_location_file():
        dock_loc_path = os.path.join(ServerConstants.get_data_dir(), "docker-location.yml")
        return dock_loc_path

    @staticmethod
    def generate_yaml_doc(run_config_object, yaml_file):
        try:
            file = open(yaml_file, 'w', encoding='utf-8')
            yaml.dump(run_config_object, file)
            file.close()
        except Exception as e:
            pass

    @staticmethod
    def exit_process(process):
        if process is None:
            return

        try:
            process.terminate()
            process.join()
            process = None
        except Exception as e:
            pass

    @staticmethod
    def exec_console_with_script(script_path, should_capture_stdout=False, should_capture_stderr=False):
        stdout_flag = subprocess.PIPE if should_capture_stdout else sys.stdout
        stderr_flag = subprocess.PIPE if should_capture_stderr else sys.stderr

        if platform.system() == 'Windows':
            script_process = subprocess.Popen(script_path, stdout=stdout_flag, stderr=stderr_flag,
                                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            script_process = subprocess.Popen(['bash', '-c', script_path], stdout=stdout_flag, stderr=stderr_flag,
                                              preexec_fn=os.setsid)

        return script_process

    @staticmethod
    def exec_console_with_shell(shell, script_path, should_capture_stdout=False, should_capture_stderr=False):
        stdout_flag = subprocess.PIPE if should_capture_stdout else sys.stdout
        stderr_flag = subprocess.PIPE if should_capture_stderr else sys.stderr

        if platform.system() == 'Windows':
            script_process = subprocess.Popen([shell, script_path], stdout=stdout_flag, stderr=stderr_flag,
                                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            script_process = subprocess.Popen([shell, script_path], stdout=stdout_flag, stderr=stderr_flag,
                                              preexec_fn=os.setsid)

        return script_process

    @staticmethod
    def exec_console_with_shell_script_list(shell_script_list, should_capture_stdout=False,
                                            should_capture_stderr=False):
        stdout_flag = subprocess.PIPE if should_capture_stdout else sys.stdout
        stderr_flag = subprocess.PIPE if should_capture_stderr else sys.stderr

        if platform.system() == 'Windows':
            script_process = subprocess.Popen(shell_script_list, stdout=stdout_flag, stderr=stderr_flag,
                                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            script_process = subprocess.Popen(shell_script_list, stdout=stdout_flag, stderr=stderr_flag,
                                              preexec_fn=os.setsid)

        return script_process

    @staticmethod
    def get_console_pipe_out_err_results(script_process):
        exec_out, exec_err = script_process.communicate()
        return script_process.returncode, exec_out, exec_err

    @staticmethod
    def get_console_sys_out_pipe_err_results(script_process):
        pipe_out, pipe_err = script_process.communicate()
        exec_out, exec_err = sys.stdout, pipe_err
        return script_process.returncode, exec_out, exec_err

    @staticmethod
    def print_console_output(script_process):
        for info in iter(script_process.stdout.readline, ""):
            print(info)

        for info in iter(script_process.stderr.readline, ""):
            print(info)

    @staticmethod
    def get_device_state_from_run_edge_state(run_edge_state):
        ret_state = ServerConstants.MSG_MLOPS_DEVICE_STATUS_IDLE
        if run_edge_state == ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE:
            ret_state = ServerConstants.MSG_MLOPS_DEVICE_STATUS_OFFLINE
        elif run_edge_state == ServerConstants.MSG_MLOPS_SERVER_STATUS_UPGRADING:
            ret_state = ServerConstants.MSG_MLOPS_DEVICE_STATUS_UPGRADING
        elif run_edge_state == ServerConstants.MSG_MLOPS_SERVER_STATUS_STARTING or \
                run_edge_state == ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING or \
                run_edge_state == ServerConstants.MSG_MLOPS_SERVER_STATUS_STOPPING:
            ret_state = ServerConstants.MSG_MLOPS_DEVICE_STATUS_RUNNING
        elif run_edge_state == ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE or \
                run_edge_state == ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED or \
                run_edge_state == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED or \
                run_edge_state == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED:
            ret_state = ServerConstants.MSG_MLOPS_DEVICE_STATUS_IDLE

        return ret_state

    @staticmethod
    def is_server_running(status):
        if status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FINISHED or \
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_KILLED or \
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_FAILED or \
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_IDLE or \
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE:
            return False

        return True
