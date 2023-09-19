import logging
import os
import platform
import signal
import subprocess
import sys
from os.path import expanduser
from urllib.parse import urlparse, unquote

import psutil
import yaml
from fedml.computing.scheduler.comm_utils import sys_utils
from ..comm_utils.run_process_utils import RunProcessUtils

from ..comm_utils.yaml_utils import load_yaml_config


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

    LOCAL_HOME_RUNNER_DIR_NAME = 'fedml-server'
    LOCAL_RUNNER_INFO_DIR_NAME = 'runner_infos'
    LOCAL_PACKAGE_HOME_DIR_NAME = "fedml_packages"

    SERVER_LOGIN_PROGRAM = "server_login.py"
    SERVER_BOOTSTRAP_LINUX_PROGRAM = "bootstrap.sh"
    SERVER_BOOTSTRAP_WIN_PROGRAM = "bootstrap.bat"

    FEDML_OTA_CMD_UPGRADE = "upgrade"
    FEDML_OTA_CMD_RESTART = "restart"

    LOCAL_SERVER_API_PORT = 40802

    LOGIN_MODE_LOCAL_INDEX = 0
    LOGIN_MODE_CLOUD_AGENT_INDEX = 1
    LOGIN_MODE_CLOUD_SERVER_INDEX = 2
    LOGIN_MODE_GPU_MASTER_SERVER_INDEX = 3

    login_role_list = ["edge_server", "cloud_agent", "cloud_server", "gpu_master_server"]

    login_index_role_map = {LOGIN_MODE_LOCAL_INDEX: login_role_list[LOGIN_MODE_LOCAL_INDEX],
                            LOGIN_MODE_CLOUD_AGENT_INDEX: login_role_list[LOGIN_MODE_CLOUD_AGENT_INDEX],
                            LOGIN_MODE_CLOUD_SERVER_INDEX: login_role_list[LOGIN_MODE_CLOUD_SERVER_INDEX],
                            LOGIN_MODE_GPU_MASTER_SERVER_INDEX: login_role_list[LOGIN_MODE_GPU_MASTER_SERVER_INDEX]}

    login_role_index_map = {login_role_list[LOGIN_MODE_LOCAL_INDEX]: LOGIN_MODE_LOCAL_INDEX,
                            login_role_list[LOGIN_MODE_CLOUD_AGENT_INDEX]: LOGIN_MODE_CLOUD_AGENT_INDEX,
                            login_role_list[LOGIN_MODE_CLOUD_SERVER_INDEX]: LOGIN_MODE_CLOUD_SERVER_INDEX,
                            login_role_list[LOGIN_MODE_GPU_MASTER_SERVER_INDEX]: LOGIN_MODE_GPU_MASTER_SERVER_INDEX}

    @staticmethod
    def get_fedml_home_dir():
        home_dir = expanduser("~")
        fedml_home_dir = os.path.join(home_dir, ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME)
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
    def get_mlops_url(config_version="release"):
        return "https://open{}.fedml.ai".format(
            "" if config_version == "release" else "-" + config_version)

    @staticmethod
    def get_job_start_url(config_version="release"):
        job_ops_url = "{}/fedmlOpsServer/api/v1/application/runApplicationFromCli".format(
            ServerConstants.get_mlops_url(config_version))
        return job_ops_url

    @staticmethod
    def get_job_list_url(config_version="release"):
        job_ops_url = "{}/fedmlOpsServer/api/v1/platform/queryJobList".format(
            ServerConstants.get_mlops_url(config_version))
        return job_ops_url

    @staticmethod
    def get_job_stop_url(config_version="release"):
        job_ops_url = "{}/fedmlOpsServer/api/v1/application/stopApplicationFromCli".format(
            ServerConstants.get_mlops_url(config_version))
        return job_ops_url

    @staticmethod
    def get_job_logs_url(config_version="release"):
        job_ops_url = "{}/fedmlOpsServer/api/v1/log/getLogsFromCli".format(
            ServerConstants.get_mlops_url(config_version))
        return job_ops_url

    @staticmethod
    def get_app_create_url(config_version="release"):
        app_url = "{}/fedmlOpsServer/api/v1/application/createApplicationFromCli".format(
            ServerConstants.get_mlops_url(config_version))
        return app_url

    @staticmethod
    def get_app_update_url(config_version="release"):
        app_url = "{}/fedmlOpsServer/api/v1/application/updateApplicationFromCli".format(
            ServerConstants.get_mlops_url(config_version))
        return app_url

    @staticmethod
    def get_heartbeat_url(config_version="release"):
        heartbeat_url = "{}/fedmlOpsServer/api/v1/cli/heartBeat".format(
            ServerConstants.get_mlops_url(config_version))
        return heartbeat_url

    @staticmethod
    def get_resource_url(config_version="release"):
        resource_url = "{}/fedmlOpsServer/api/v1/cli/resourceType".format(
            ServerConstants.get_mlops_url(config_version))
        return resource_url

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
            info_file_prefix="user-process")

    @staticmethod
    def save_learning_process(run_id, learning_id):
        RunProcessUtils.save_run_process(
            run_id, learning_id, ServerConstants.get_data_dir(), ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix="user-process")

    @staticmethod
    def cleanup_bootstrap_process(run_id):
        RunProcessUtils.cleanup_run_process(
            run_id, ServerConstants.get_data_dir(), ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix="bootstrap-process")

    @staticmethod
    def save_bootstrap_process(run_id, process_id):
        RunProcessUtils.save_run_process(
            run_id, process_id, ServerConstants.get_data_dir(), ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            info_file_prefix="bootstrap-process")

    @staticmethod
    def save_runner_infos(unique_device_id, edge_id, run_id=None):
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
            script_process = subprocess.Popen("dir", stdout=stdout_flag, stderr=stderr_flag,
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

    @staticmethod
    def get_filename_and_extension(url):
        parsed_url = urlparse(unquote(url))
        path = parsed_url.path
        filename = os.path.basename(path)
        filename_without_extension, file_extension = os.path.splitext(filename)
        return filename, filename_without_extension, file_extension
