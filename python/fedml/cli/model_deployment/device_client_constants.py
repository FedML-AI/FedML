import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import urllib
import zipfile
from os.path import expanduser

import psutil
import yaml
from ...cli.comm_utils.yaml_utils import load_yaml_config


class ClientConstants(object):
    MSG_MLOPS_CLIENT_STATUS_OFFLINE = "OFFLINE"
    MSG_MLOPS_CLIENT_STATUS_IDLE = "IDLE"
    MSG_MLOPS_CLIENT_STATUS_UPGRADING = "UPGRADING"
    MSG_MLOPS_CLIENT_STATUS_QUEUED = "QUEUED"
    MSG_MLOPS_CLIENT_STATUS_INITIALIZING = "INITIALIZING"
    MSG_MLOPS_CLIENT_STATUS_TRAINING = "TRAINING"
    MSG_MLOPS_CLIENT_STATUS_RUNNING = "RUNNING"
    MSG_MLOPS_CLIENT_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_CLIENT_STATUS_KILLED = "KILLED"
    MSG_MLOPS_CLIENT_STATUS_FAILED = "FAILED"
    MSG_MLOPS_CLIENT_STATUS_FINISHED = "FINISHED"

    MSG_MLOPS_SERVER_DEVICE_STATUS_OFFLINE = "OFFLINE"
    MSG_MLOPS_SERVER_DEVICE_STATUS_IDLE = "IDLE"
    MSG_MLOPS_SERVER_DEVICE_STATUS_STARTING = "STARTING"
    MSG_MLOPS_SERVER_DEVICE_STATUS_RUNNING = "RUNNING"
    MSG_MLOPS_SERVER_DEVICE_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_SERVER_DEVICE_STATUS_KILLED = "KILLED"
    MSG_MLOPS_SERVER_DEVICE_STATUS_FAILED = "FAILED"
    MSG_MLOPS_SERVER_DEVICE_STATUS_FINISHED = "FINISHED"

    LOCAL_HOME_RUNNER_DIR_NAME = 'fedml-client'
    LOCAL_RUNNER_INFO_DIR_NAME = 'runner_infos'
    LOCAL_PACKAGE_HOME_DIR_NAME = "fedml_packages"

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

    FEDML_OTA_CMD_UPGRADE = "upgrade"
    FEDML_OTA_CMD_RESTART = "restart"

    LOGIN_MODE_ON_PREMISE_INDEX = 0
    LOGIN_MODE_FEDML_CLOUD_INDEX = 1
    LOGIN_MODE_PUBLIC_CLOUD_INDEX = 2
    login_role_list = ["md.on_premise_device", "md.fedml_cloud_device", "md.pubic_cloud_device"]

    @staticmethod
    def get_fedml_home_dir():
        home_dir = expanduser("~")
        fedml_home_dir = os.path.join(home_dir, ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME)
        return fedml_home_dir

    @staticmethod
    def get_log_file_dir():
        log_file_dir = os.path.join(ClientConstants.get_fedml_home_dir(), "fedml", "logs")
        return log_file_dir

    @staticmethod
    def get_data_dir():
        data_dir = os.path.join(ClientConstants.get_fedml_home_dir(), "fedml", "data")
        return data_dir

    @staticmethod
    def get_package_download_dir():
        package_download_dir = os.path.join(ClientConstants.get_fedml_home_dir(),
                                            ClientConstants.LOCAL_PACKAGE_HOME_DIR_NAME)
        return package_download_dir

    @staticmethod
    def get_package_unzip_dir():
        package_unzip_dir = ClientConstants.get_package_download_dir()
        return package_unzip_dir

    @staticmethod
    def get_package_run_dir(package_name):
        package_file_no_extension = str(package_name).split('.')[0]
        package_run_dir = os.path.join(ClientConstants.get_package_unzip_dir(),
                                       package_file_no_extension)
        return package_run_dir

    @staticmethod
    def get_model_dir():
        model_file_dir = os.path.join(ClientConstants.get_fedml_home_dir(), "fedml", "models")
        return model_file_dir

    @staticmethod
    def get_model_package_dir():
        model_packages_dir = os.path.join(ClientConstants.get_fedml_home_dir(), "fedml", "model_packages")
        return model_packages_dir

    @staticmethod
    def get_model_cache_dir():
        model_cache_dir = os.path.join(ClientConstants.get_fedml_home_dir(), "fedml", "model_cache")
        return model_cache_dir

    @staticmethod
    def get_k8s_master_host_dir(current_dir):
        if not ClientConstants.is_running_on_k8s():
            return current_dir

        if str(current_dir).startswith(ClientConstants.K8S_DEPLOYMENT_MASTER_MOUNT_HOME_DIR):
            return str(current_dir).replace(ClientConstants.K8S_DEPLOYMENT_MASTER_MOUNT_HOME_DIR,
                                            ClientConstants.K8S_DEPLOYMENT_MASTER_HOST_HOME_DIR)
        return current_dir

    @staticmethod
    def get_k8s_slave_host_dir(current_dir):
        if not ClientConstants.is_running_on_k8s():
            return current_dir

        if str(current_dir).startswith(ClientConstants.K8S_DEPLOYMENT_SLAVE_MOUNT_HOME_DIR):
            return str(current_dir).replace(ClientConstants.K8S_DEPLOYMENT_SLAVE_MOUNT_HOME_DIR,
                                            ClientConstants.K8S_DEPLOYMENT_SLAVE_HOST_HOME_DIR)
        return current_dir

    @staticmethod
    def is_running_on_k8s():
        running_source = os.getenv(ClientConstants.FEDML_RUNNING_SOURCE_ENV_NAME, default=None)
        if running_source is not None and running_source == ClientConstants.FEDML_RUNNING_SOURCE_ENV_VALUE_K8S:
            return True
        return False

    @staticmethod
    def get_model_serving_dir():
        model_file_dir = os.path.join(ClientConstants.get_fedml_home_dir(), "fedml", "models_serving")
        return model_file_dir

    @staticmethod
    def get_model_infer_data_dir():
        model_infer_data_dir = os.path.join(ClientConstants.get_fedml_home_dir(), "fedml", "models_infer_data")
        return model_infer_data_dir

    @staticmethod
    def get_model_ops_list_url(config_version="release"):
        model_ops_url = "{}/api/v1/model/listFromCli".format(ClientConstants.get_model_ops_url(config_version))
        return model_ops_url

    @staticmethod
    def get_model_ops_upload_url(config_version="release"):
        model_ops_url = "{}/api/v1/model/createFromCli".format(ClientConstants.get_model_ops_url(config_version))
        return model_ops_url

    @staticmethod
    def get_model_ops_url(config_version="release"):
        return "https://model{}.fedml.ai/fedmlModelServer".format(
            "" if config_version == "release" else "-" + config_version)

    @staticmethod
    def get_model_ops_deployment_url(config_version="release"):
        model_ops_url = "{}/api/v1/endpoint/createFromCli".format(ClientConstants.get_model_ops_url(config_version))
        return model_ops_url

    @staticmethod
    def get_running_model_name(end_point_id, model_id, model_name, model_version):
        running_model_name = "model_{}_{}_{}_{}".format(end_point_id, model_id, model_name, model_version)
        running_model_name = running_model_name.replace(' ', '-')
        running_model_name = running_model_name.replace(':', '-')
        return running_model_name

    @staticmethod
    def remove_deployment(end_point_id, model_id, model_name, model_version):
        running_model_name = ClientConstants.get_running_model_name(end_point_id, model_id, model_name, model_version)
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name,
                                 ClientConstants.FEDML_CONVERTED_MODEL_DIR_NAME)
        model_dir_list = os.listdir(model_dir)
        for dir_item in model_dir_list:
            if not dir_item.startswith(running_model_name):
                continue
            logging.info("remove model file {}.".format(dir_item))
            model_file_path = os.path.join(model_dir, dir_item)
            shutil.rmtree(model_file_path, ignore_errors=True)
            os.system("sudo rm -Rf {}".format(model_file_path))

        model_serving_dir = ClientConstants.get_model_serving_dir()
        if not os.path.exists(model_serving_dir):
            return False
        serving_dir_list = os.listdir(model_serving_dir)
        for dir_item in serving_dir_list:
            if not dir_item.startswith(running_model_name):
                continue
            logging.info("remove model serving file {}.".format(dir_item))
            model_file_path = os.path.join(model_serving_dir, dir_item)
            shutil.rmtree(model_file_path, ignore_errors=True)
            os.system("sudo rm -Rf {}".format(model_file_path))

        return True

    @staticmethod
    def get_local_ip():
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        conn = s.connect(('8.8.8.8', 53))
        ip = s.getsockname()[0]
        s.close()
        return ip

    @staticmethod
    def check_network_port_is_opened(port):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(('localhost', int(port)))
            s.settimeout(1)
            s.shutdown(2)
            return True
        except:
            return False

    @staticmethod
    def check_process_is_running(process_id):
        for proc in psutil.process_iter():
            try:
                if process_id == proc.pid:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    @staticmethod
    def unzip_file(zip_file, unzip_file_path):
        result = False
        if zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zipf:
                zipf.extractall(unzip_file_path)
                result = True

        return result

    @staticmethod
    def retrieve_and_unzip_package(package_url, package_name, local_package_file, unzip_package_path):
        if not os.path.exists(local_package_file):
            urllib.request.urlretrieve(package_url, local_package_file)
        try:
            shutil.rmtree(
                os.path.join(unzip_package_path, package_name), ignore_errors=True
            )
        except Exception as e:
            pass
        ClientConstants.unzip_file(local_package_file, unzip_package_path)
        unzip_package_path = os.path.join(unzip_package_path, package_name)
        return unzip_package_path

    @staticmethod
    def cleanup_run_process():
        try:
            local_pkg_data_dir = ClientConstants.get_data_dir()
            process_id_file = os.path.join(local_pkg_data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                           "runner-sub-process.id")
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
            ClientConstants.generate_yaml_doc(yaml_object, process_id_file)
        except Exception as e:
            pass

    @staticmethod
    def save_run_process(process_id):
        try:
            local_pkg_data_dir = ClientConstants.get_data_dir()
            process_id_file = os.path.join(local_pkg_data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                           "runner-sub-process.id")
            yaml_object = {}
            yaml_object['process_id'] = process_id
            ClientConstants.generate_yaml_doc(yaml_object, process_id_file)
        except Exception as e:
            pass

    @staticmethod
    def cleanup_learning_process():
        try:
            local_pkg_data_dir = ClientConstants.get_data_dir()
            process_id_file = os.path.join(local_pkg_data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                           "runner-learning-process.id")
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
            ClientConstants.generate_yaml_doc(yaml_object, process_id_file)
        except Exception as e:
            pass

    @staticmethod
    def save_learning_process(learning_id):
        try:
            local_pkg_data_dir = ClientConstants.get_data_dir()
            process_id_file = os.path.join(local_pkg_data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                           "runner-learning-process.id")
            yaml_object = {}
            yaml_object['process_id'] = learning_id
            ClientConstants.generate_yaml_doc(yaml_object, process_id_file)
        except Exception as e:
            pass

    @staticmethod
    def save_runner_infos(unique_device_id, edge_id, run_id=None):
        local_pkg_data_dir = ClientConstants.get_data_dir()
        try:
            os.makedirs(local_pkg_data_dir)
        except Exception as e:
            pass
        try:
            os.makedirs(os.path.join(local_pkg_data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME))
        except Exception as e:
            pass

        runner_info_file = os.path.join(local_pkg_data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                        "runner_infos.yaml")
        running_info = dict()
        running_info["unique_device_id"] = str(unique_device_id)
        running_info["edge_id"] = str(edge_id)
        running_info["run_id"] = run_id
        ClientConstants.generate_yaml_doc(running_info, runner_info_file)

    @staticmethod
    def save_training_infos(edge_id, training_status):
        local_pkg_data_dir = ClientConstants.get_data_dir()
        try:
            os.makedirs(local_pkg_data_dir)
        except Exception as e:
            pass
        try:
            os.makedirs(os.path.join(local_pkg_data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME))
        except Exception as e:
            pass

        training_info_file = os.path.join(local_pkg_data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                          "training_infos.yaml")
        training_info = dict()
        training_info["edge_id"] = edge_id
        training_info["training_status"] = str(training_status)
        ClientConstants.generate_yaml_doc(training_info, training_info_file)

    @staticmethod
    def get_training_infos():
        local_pkg_data_dir = ClientConstants.get_data_dir()
        training_info_file = os.path.join(local_pkg_data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                          "training_infos.yaml")
        training_info = dict()
        training_info["edge_id"] = 0
        training_info["training_status"] = "INITIALIZING"
        try:
            training_info = load_yaml_config(training_info_file)
        except Exception as e:
            pass
        return training_info

    @staticmethod
    def get_docker_location_file():
        dock_loc_path = os.path.join(ClientConstants.get_data_dir(), "docker-location.yml")
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
    def exec_console_with_script(script_path, should_capture_stdout=False, should_capture_stderr=False, no_sys_out_err=False):
        stdout_flag = subprocess.PIPE if should_capture_stdout else sys.stdout
        stderr_flag = subprocess.PIPE if should_capture_stderr else sys.stderr

        if platform.system() == 'Windows':
            if no_sys_out_err:
                script_process = subprocess.Popen(script_path)
            else:
                script_process = subprocess.Popen(script_path, stdout=stdout_flag, stderr=stderr_flag)
        else:
            if no_sys_out_err:
                script_process = subprocess.Popen(['bash', '-c', script_path])
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
    def exec_console_with_shell_script_list(shell_script_list, should_capture_stdout=False,
                                            should_capture_stderr=False):
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


if __name__ == "__main__":
    ignore = "*test*,abc*"
    ignore = tuple(ignore.split(','))
    shutil.rmtree("/Users/alexliang/fedml-test/examples2", ignore_errors=True)
    shutil.copytree("/Users/alexliang/fedml-test/examples",
                    "/Users/alexliang/fedml-test/examples2",
                    ignore=shutil.ignore_patterns(*ignore))

    script_process = ClientConstants.exec_console_with_shell_script_list(
        ['sh', '-c', "while [ 1 = 1 ]; do echo 'hello'; sleep 1; done "])
    ClientConstants.print_console_output(script_process)
    ret_code, out, err = ClientConstants.get_console_pipe_out_err_results(script_process)
    print("script process {}".format(script_process.pid))
