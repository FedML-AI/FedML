
import json
import os
import platform
import signal
import subprocess
import sys
from os.path import expanduser

import psutil
import yaml
from ...cli.comm_utils.yaml_utils import load_yaml_config


class ServerConstants(object):
    MSG_MLOPS_SERVER_STATUS_OFFLINE = "OFFLINE"
    MSG_MLOPS_SERVER_STATUS_IDLE = "IDLE"
    MSG_MLOPS_SERVER_STATUS_STARTING = "STARTING"
    MSG_MLOPS_SERVER_STATUS_RUNNING = "RUNNING"
    MSG_MLOPS_SERVER_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_SERVER_STATUS_KILLED = "KILLED"
    MSG_MLOPS_SERVER_STATUS_FAILED = "FAILED"
    MSG_MLOPS_SERVER_STATUS_FINISHED = "FINISHED"

    LOCAL_HOME_RUNNER_DIR_NAME = 'fedml-server'
    LOCAL_RUNNER_INFO_DIR_NAME = 'runner_infos'
    LOCAL_PACKAGE_HOME_DIR_NAME = "fedml_packages"

    FEDML_OTA_CMD_UPGRADE = "upgrade"
    FEDML_OTA_CMD_RESTART = "restart"

    # Constants for models
    K8S_DEPLOYMENT_MASTER_HOST_HOME_DIR = "/home/fedml-server"
    K8S_DEPLOYMENT_SLAVE_HOST_HOME_DIR = "/home/fedml-client"
    K8S_DEPLOYMENT_MASTER_MOUNT_HOME_DIR = "/home/fedml/fedml-server"
    K8S_DEPLOYMENT_SLAVE_MOUNT_HOME_DIR = "/home/fedml/fedml-client"

    INFERENCE_HTTP_PORT = 8000
    INFERENCE_GRPC_PORT = 8001
    INFERENCE_METRIC_PORT = 8002

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

    @staticmethod
    def get_fedml_home_dir():
        home_dir = expanduser("~")
        fedml_home_dir = os.path.join(home_dir, ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME)
        return fedml_home_dir

    @staticmethod
    def get_log_file_dir():
        log_file_dir = os.path.join(ServerConstants.get_fedml_home_dir(), "fedml", "logs")
        return log_file_dir

    @staticmethod
    def get_data_dir():
        data_dir = os.path.join(ServerConstants.get_fedml_home_dir(), "fedml", "data")
        return data_dir

    @staticmethod
    def get_package_download_dir():
        package_download_dir = os.path.join(ServerConstants.get_fedml_home_dir(), ServerConstants.LOCAL_PACKAGE_HOME_DIR_NAME)
        return package_download_dir

    @staticmethod
    def get_package_unzip_dir(run_id, package_url):
        package_unzip_dir_name = "run_{}_{}".format(str(run_id),
                                                    str(os.path.basename(package_url)).split('.')[0])
        package_unzip_dir = os.path.join(ServerConstants.get_package_download_dir(),
                                         package_unzip_dir_name)
        return package_unzip_dir

    @staticmethod
    def get_package_run_dir(run_id, package_url, package_name):
        package_file_no_extension = str(package_name).split('.')[0]
        package_run_dir = os.path.join(ServerConstants.get_package_unzip_dir(run_id, package_url),
                                       package_file_no_extension)
        return package_run_dir

    @staticmethod
    def get_model_dir():
        model_file_dir = os.path.join(ServerConstants.get_fedml_home_dir(), "fedml", "models")
        return model_file_dir

    @staticmethod
    def get_model_package_dir():
        model_packages_dir = os.path.join(ServerConstants.get_fedml_home_dir(), "fedml", "model_packages")
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
        return "https://model{}.fedml.ai/fedmlModelServer".format(
            "" if config_version == "release" else "-" + config_version)

    @staticmethod
    def get_model_ops_deployment_url(config_version="release"):
        model_ops_url = "{}/api/v1/endpoint/createFromCli".format(ServerConstants.get_model_ops_url(config_version))
        return model_ops_url

    @staticmethod
    def get_running_model_name(end_point_id, model_id, model_name, model_version):
        running_model_name = "model_{}_{}_{}_{}".format(end_point_id, model_id, model_name, model_version)
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
    def cleanup_run_process():
        try:
            home_dir = expanduser("~")
            local_pkg_data_dir = ServerConstants.get_data_dir()
            process_id_file = os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME, "runner-sub-process.id")
            process_info = load_yaml_config(process_id_file)
            process_ids_str = process_info.get('process_id', '[]')
            process_ids = json.loads(process_ids_str)
            for process_id in process_ids:
                try:
                    process = psutil.Process(process_id)
                    for sub_process in process.children():
                        os.kill(sub_process.pid, signal.SIGTERM)

                    if process is not None:
                        os.kill(process.pid, signal.SIGTERM)
                except Exception as e:
                    pass

            yaml_object = {}
            yaml_object['process_id'] = '[]'
            ServerConstants.generate_yaml_doc(yaml_object, process_id_file)
        except Exception as e:
            pass

    @staticmethod
    def save_run_process(process_id):
        try:
            home_dir = expanduser("~")
            local_pkg_data_dir = ServerConstants.get_data_dir()
            process_id_file = os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME, "runner-sub-process.id")
            process_ids = []
            if os.path.exists(process_id_file) is True:
                yaml_object = load_yaml_config(process_id_file)
                process_ids_str = yaml_object.get('process_id', '[]')
                process_ids = json.loads(process_ids_str)
            process_ids.append(process_id)
            yaml_object = {}
            yaml_object['process_id'] = str(process_ids)
            ServerConstants.generate_yaml_doc(yaml_object, process_id_file)
        except Exception as e:
            pass

    @staticmethod
    def cleanup_learning_process():
        try:
            home_dir = expanduser("~")
            local_pkg_data_dir = ServerConstants.get_data_dir()
            process_id_file = os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME, "runner-learning-process.id")
            process_info = load_yaml_config(process_id_file)
            process_id = process_info.get('process_id', None)
            if process_id is not None:
                try:
                    process = psutil.Process(process_id)
                    for sub_process in process.children():
                        os.kill(sub_process.pid, signal.SIGTERM)

                    if process is not None:
                        os.kill(process.pid, signal.SIGTERM)
                except Exception as e:
                    pass
            yaml_object = {}
            yaml_object['process_id'] = -1
            ServerConstants.generate_yaml_doc(yaml_object, process_id_file)
        except Exception as e:
            pass

    @staticmethod
    def save_learning_process(learning_id):
        try:
            home_dir = expanduser("~")
            local_pkg_data_dir = ServerConstants.get_data_dir()
            process_id_file = os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME, "runner-learning-process.id")
            yaml_object = {}
            yaml_object['process_id'] = learning_id
            ServerConstants.generate_yaml_doc(yaml_object, process_id_file)
        except Exception as e:
            pass

    @staticmethod
    def save_runner_infos(unique_device_id, edge_id, run_id=None):
        home_dir = expanduser("~")
        local_pkg_data_dir = ServerConstants.get_data_dir()
        try:
            os.makedirs(local_pkg_data_dir)
        except Exception as e:
            pass
        try:
            os.makedirs(os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME))
        except Exception as e:
            pass

        runner_info_file = os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME, "runner_infos.yaml")
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
            script_process = subprocess.Popen(script_path, stdout=stdout_flag, stderr=stderr_flag)
        else:
            script_process = subprocess.Popen(['bash', '-c', script_path], stdout=stdout_flag, stderr=stderr_flag)

        return script_process

    @staticmethod
    def exec_console_with_shell(shell, script_path, should_capture_stdout=False, should_capture_stderr=False):
        stdout_flag = subprocess.PIPE if should_capture_stdout else sys.stdout
        stderr_flag = subprocess.PIPE if should_capture_stderr else sys.stderr

        script_process = subprocess.Popen([shell, script_path], stdout=stdout_flag, stderr=stderr_flag)

        return script_process

    @staticmethod
    def exec_console_with_shell_script_list(shell_script_list, should_capture_stdout=False, should_capture_stderr=False):
        stdout_flag = subprocess.PIPE if should_capture_stdout else sys.stdout
        stderr_flag = subprocess.PIPE if should_capture_stderr else sys.stderr

        script_process = subprocess.Popen(shell_script_list, stdout=stdout_flag, stderr=stderr_flag)

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
