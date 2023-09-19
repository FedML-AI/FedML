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
from urllib.parse import urlparse, unquote

import psutil
import yaml
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.comm_utils.run_process_utils import RunProcessUtils

from ..comm_utils.yaml_utils import load_yaml_config


class ClientConstants(object):
    MSG_MLOPS_CLIENT_STATUS_OFFLINE = "OFFLINE"
    MSG_MLOPS_CLIENT_STATUS_IDLE = "IDLE"
    MSG_MLOPS_CLIENT_STATUS_UPGRADING = "UPGRADING"
    MSG_MLOPS_CLIENT_STATUS_QUEUED = "QUEUED"
    MSG_MLOPS_CLIENT_STATUS_INITIALIZING = "INITIALIZING"
    MSG_MLOPS_CLIENT_STATUS_TRAINING = "TRAINING"
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

    LOCAL_HOME_RUNNER_DIR_NAME = 'fedml-client'
    LOCAL_RUNNER_INFO_DIR_NAME = 'runner_infos'
    LOCAL_PACKAGE_HOME_DIR_NAME = "fedml_packages"

    CLIENT_LOGIN_PROGRAM = "client_login.py"
    CLIENT_BOOTSTRAP_LINUX_PROGRAM = "bootstrap.sh"
    CLIENT_BOOTSTRAP_WIN_PROGRAM = "bootstrap.bat"

    CLIENT_SHELL_BASH = "bash"
    CLIENT_SHELL_PS = "powershell"
    PLATFORM_WINDOWS = "Windows"

    FEDML_OTA_CMD_UPGRADE = "upgrade"
    FEDML_OTA_CMD_RESTART = "restart"

    LOCAL_CLIENT_API_PORT = 40800

    LOGIN_MODE_CLIEN_INDEX = 0
    LOGIN_MODE_EDGE_SIMULATOR_INDEX = 1
    LOGIN_MODE_GPU_SUPPLIER_INDEX = 2

    login_role_list = ["client", "edge_simulator", "gpu_supplier"]

    login_index_role_map = {LOGIN_MODE_CLIEN_INDEX: login_role_list[LOGIN_MODE_CLIEN_INDEX],
                            LOGIN_MODE_EDGE_SIMULATOR_INDEX: login_role_list[LOGIN_MODE_EDGE_SIMULATOR_INDEX],
                            LOGIN_MODE_GPU_SUPPLIER_INDEX: login_role_list[LOGIN_MODE_GPU_SUPPLIER_INDEX]}

    login_role_index_map = {login_role_list[LOGIN_MODE_CLIEN_INDEX]: LOGIN_MODE_CLIEN_INDEX,
                            login_role_list[LOGIN_MODE_EDGE_SIMULATOR_INDEX]: LOGIN_MODE_EDGE_SIMULATOR_INDEX,
                            login_role_list[LOGIN_MODE_GPU_SUPPLIER_INDEX]: LOGIN_MODE_GPU_SUPPLIER_INDEX}

    @staticmethod
    def get_fedml_home_dir():
        home_dir = expanduser("~")
        fedml_home_dir = os.path.join(home_dir, ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME)
        if not os.path.exists(fedml_home_dir):
            os.makedirs(fedml_home_dir, exist_ok=True)
        return fedml_home_dir

    @staticmethod
    def get_log_file_dir():
        log_file_dir = os.path.join(ClientConstants.get_fedml_home_dir(), "fedml", "logs")
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir, exist_ok=True)
        return log_file_dir

    @staticmethod
    def get_data_dir():
        data_dir = os.path.join(ClientConstants.get_fedml_home_dir(), "fedml", "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        return data_dir

    @staticmethod
    def get_package_download_dir():
        package_download_dir = os.path.join(ClientConstants.get_fedml_home_dir(),
                                            ClientConstants.LOCAL_PACKAGE_HOME_DIR_NAME)
        if not os.path.exists(package_download_dir):
            os.makedirs(package_download_dir, exist_ok=True)
        return package_download_dir

    @staticmethod
    def get_package_unzip_dir():
        package_unzip_dir = ClientConstants.get_package_download_dir()
        if not os.path.exists(package_unzip_dir):
            os.makedirs(package_unzip_dir, exist_ok=True)
        return package_unzip_dir

    @staticmethod
    def get_package_run_dir(package_name):
        package_file_no_extension = str(package_name).split('.')[0]
        package_run_dir = os.path.join(ClientConstants.get_package_unzip_dir(),
                                       package_file_no_extension)
        if not os.path.exists(package_run_dir):
            os.makedirs(package_run_dir, exist_ok=True)
        return package_run_dir

    @staticmethod
    def get_model_cache_dir():
        model_cache_dir = os.path.join(ClientConstants.get_fedml_home_dir(), "fedml", "model_cache")
        if not os.path.exists(model_cache_dir):
            os.makedirs(model_cache_dir, exist_ok=True)
        return model_cache_dir

    @staticmethod
    def get_database_dir():
        database_dir = os.path.join(ClientConstants.get_data_dir(), "database")
        if not os.path.exists(database_dir):
            os.makedirs(database_dir, exist_ok=True)
        return database_dir

    @staticmethod
    def cleanup_run_process(run_id):
        RunProcessUtils.cleanup_run_process(
            run_id, ClientConstants.get_data_dir(), ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)

    @staticmethod
    def save_run_process(run_id, process_id):
        RunProcessUtils.save_run_process(
            run_id, process_id, ClientConstants.get_data_dir(), ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)

    @staticmethod
    def cleanup_learning_process(run_id):
        RunProcessUtils.cleanup_run_process(
            run_id, ClientConstants.get_data_dir(), ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix="user-process")

    @staticmethod
    def save_learning_process(run_id, learning_id):
        RunProcessUtils.save_run_process(
            run_id, learning_id, ClientConstants.get_data_dir(), ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix="user-process")

    @staticmethod
    def cleanup_bootstrap_process(run_id):
        RunProcessUtils.cleanup_run_process(
            run_id, ClientConstants.get_data_dir(), ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix="bootstrap-process")

    @staticmethod
    def save_bootstrap_process(run_id, process_id):
        RunProcessUtils.save_run_process(
            run_id, process_id, ClientConstants.get_data_dir(), ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix="bootstrap-process")

    @staticmethod
    def save_runner_infos(unique_device_id, edge_id, run_id=None):
        local_pkg_data_dir = ClientConstants.get_data_dir()
        os.makedirs(local_pkg_data_dir, exist_ok=True)
        os.makedirs(os.path.join(local_pkg_data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME), exist_ok=True)

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
        os.makedirs(local_pkg_data_dir, exist_ok=True)
        os.makedirs(os.path.join(local_pkg_data_dir, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME), exist_ok=True)

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
    def execute_commands_with_live_logs(cmds, join='&&', should_write_log_file=True,
                                        callback=None, error_processor=None):
        error_list = list()
        with subprocess.Popen(join.join(cmds),
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as sp:
            if callback is not None:
                callback(sp.pid)

            if should_write_log_file:
                with os.fdopen(sys.stdout.fileno(), 'wb', closefd=False) as stdout:
                    for line in sp.stdout:
                        line_str = sys_utils.decode_byte_str(line)
                        stdout.write(line)
                        stdout.flush()
                        logging.info(line_str)
            with os.fdopen(sys.stderr.fileno(), 'wb', closefd=False) as stderr:
                for line in sp.stderr:
                    line_str = sys_utils.decode_byte_str(line)
                    if should_write_log_file:
                        stderr.write(line)
                        stderr.flush()
                        logging.error(line_str)
                    error_list.append(line_str)

                if error_processor is not None and len(error_list) > 0:
                    error_processor(error_list)

            return sp, error_list
        return None, error_list

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
        ret_state = ClientConstants.MSG_MLOPS_DEVICE_STATUS_IDLE
        if run_edge_state == ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE:
            ret_state = ClientConstants.MSG_MLOPS_DEVICE_STATUS_OFFLINE
        elif run_edge_state == ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING:
            ret_state = ClientConstants.MSG_MLOPS_DEVICE_STATUS_UPGRADING
        elif run_edge_state == ClientConstants.MSG_MLOPS_CLIENT_STATUS_QUEUED or \
                run_edge_state == ClientConstants.MSG_MLOPS_CLIENT_STATUS_INITIALIZING or \
                run_edge_state == ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING or \
                run_edge_state == ClientConstants.MSG_MLOPS_CLIENT_STATUS_STOPPING:
            ret_state = ClientConstants.MSG_MLOPS_DEVICE_STATUS_RUNNING
        elif run_edge_state == ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE or \
                run_edge_state == ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED or \
                run_edge_state == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                run_edge_state == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED:
            ret_state = ClientConstants.MSG_MLOPS_DEVICE_STATUS_IDLE

        return ret_state

    @staticmethod
    def is_client_running(status):
        if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED or \
                status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED or \
                status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_IDLE or \
                status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE:
            return False

        return True

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
    def get_filename_and_extension(url):
        parsed_url = urlparse(unquote(url))
        path = parsed_url.path
        filename = os.path.basename(path)
        filename_without_extension, file_extension = os.path.splitext(filename)
        return filename, filename_without_extension, file_extension


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
