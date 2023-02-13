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
    MSG_MLOPS_SERVER_STATUS_UPGRADING = "UPGRADING"
    MSG_MLOPS_SERVER_STATUS_STARTING = "STARTING"
    MSG_MLOPS_SERVER_STATUS_RUNNING = "RUNNING"
    MSG_MLOPS_SERVER_STATUS_STOPPING = "STOPPING"
    MSG_MLOPS_SERVER_STATUS_KILLED = "KILLED"
    MSG_MLOPS_SERVER_STATUS_FAILED = "FAILED"
    MSG_MLOPS_SERVER_STATUS_FINISHED = "FINISHED"

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

    LOCAL_HOME_RUNNER_DIR_NAME = 'fedml-server'
    LOCAL_RUNNER_INFO_DIR_NAME = 'runner_infos'
    LOCAL_PACKAGE_HOME_DIR_NAME = "fedml_packages"

    FEDML_OTA_CMD_UPGRADE = "upgrade"
    FEDML_OTA_CMD_RESTART = "restart"

    LOCAL_SERVER_API_PORT = 40802

    LOGIN_MODE_LOCAL_INDEX = 0
    LOGIN_MODE_CLOUD_AGENT_INDEX = 1
    LOGIN_MODE_CLOUD_SERVER_INDEX = 2
    login_role_list = ["edge_server", "cloud_agent", "cloud_server"]

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
        package_download_dir = os.path.join(ServerConstants.get_fedml_home_dir(),
                                            ServerConstants.LOCAL_PACKAGE_HOME_DIR_NAME)
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
    def get_database_dir():
        database_dir = os.path.join(ServerConstants.get_data_dir(), "database")
        return database_dir

    @staticmethod
    def get_mlops_url(config_version="release"):
        return "https://open{}.fedml.ai".format(
            "" if config_version == "release" else "-" + config_version)

    @staticmethod
    def get_job_start_url(config_version="release"):
        job_ops_url = "{}/fedmlOpsServer/api/v1/application/runApplicationFromCli".format(
            ServerConstants.get_mlops_url(config_version))
        return job_ops_url

    @staticmethod
    def cleanup_run_process():
        try:
            home_dir = expanduser("~")
            local_pkg_data_dir = ServerConstants.get_data_dir()
            process_id_file = os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                           "runner-sub-process.id")
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
            process_id_file = os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                           "runner-sub-process.id")
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
            process_id_file = os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
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
            ServerConstants.generate_yaml_doc(yaml_object, process_id_file)
        except Exception as e:
            pass

    @staticmethod
    def save_learning_process(learning_id):
        try:
            home_dir = expanduser("~")
            local_pkg_data_dir = ServerConstants.get_data_dir()
            process_id_file = os.path.join(local_pkg_data_dir, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
                                           "runner-learning-process.id")
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
