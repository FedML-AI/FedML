import logging
import os
import platform
import requests
from ..comm_utils import subprocess_with_live_logs
import subprocess
import sys
from os.path import expanduser
from urllib.parse import urlparse, unquote

import yaml
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.core.mlops.mlops_configs import MLOpsConfigs
from ..comm_utils.constants import SchedulerConstants
from ..comm_utils.run_process_utils import RunProcessUtils

import fedml


class ServerConstants(object):
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

    FEDML_PARENT_PID_FILE = "fedml_parent_pid"

    LOCAL_SERVER_API_PORT = 40802

    LOGIN_MODE_LOCAL_INDEX = 0
    LOGIN_MODE_CLOUD_AGENT_INDEX = 1
    LOGIN_MODE_CLOUD_SERVER_INDEX = 2
    LOGIN_MODE_GPU_MASTER_SERVER_INDEX = 3

    ''' 
    "edge_server":  
        (on premise server, used by fl) , more detailed name: master_agent_on_prem_for_fl
    "gpu_master_server":  
        (on premise server, used by launch), more detailed name: master_agent_on_prem_for_general_launch
    "cloud_server": 
        (public server instance), more detailed name: master_agent_cloud_for_fl、master_agent_cloud_for_general_launch
    "cloud-agent": 
        (public server container which will initiate a public server instance when the user selects a 
        FedML Public Cloud Server as the aggregation sever or launch server to start a run) , 
        more detailed name: master_agent_cloud_proxy_for_fl、master_agent_cloud_proxy_for_general_launch
    '''
    login_role_list = ["edge_server", "cloud_agent", "cloud_server", "gpu_master_server"]

    login_index_role_map = {LOGIN_MODE_LOCAL_INDEX: login_role_list[LOGIN_MODE_LOCAL_INDEX],
                            LOGIN_MODE_CLOUD_AGENT_INDEX: login_role_list[LOGIN_MODE_CLOUD_AGENT_INDEX],
                            LOGIN_MODE_CLOUD_SERVER_INDEX: login_role_list[LOGIN_MODE_CLOUD_SERVER_INDEX],
                            LOGIN_MODE_GPU_MASTER_SERVER_INDEX: login_role_list[LOGIN_MODE_GPU_MASTER_SERVER_INDEX]}

    login_role_index_map = {login_role_list[LOGIN_MODE_LOCAL_INDEX]: LOGIN_MODE_LOCAL_INDEX,
                            login_role_list[LOGIN_MODE_CLOUD_AGENT_INDEX]: LOGIN_MODE_CLOUD_AGENT_INDEX,
                            login_role_list[LOGIN_MODE_CLOUD_SERVER_INDEX]: LOGIN_MODE_CLOUD_SERVER_INDEX,
                            login_role_list[LOGIN_MODE_GPU_MASTER_SERVER_INDEX]: LOGIN_MODE_GPU_MASTER_SERVER_INDEX}

    API_HEADERS = {'Content-Type': 'application/json', 'Connection': 'close'}

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
    def get_mlops_url():
        url = fedml._get_backend_service()
        return url

    @staticmethod
    def get_run_start_url():
        run_ops_url = "{}/fedmlOpsServer/api/v1/application/runApplicationFromCli".format(
            ServerConstants.get_mlops_url())
        return run_ops_url

    @staticmethod
    def get_run_list_url():
        run_ops_url = "{}/fedmlOpsServer/api/v1/platform/queryJobList".format(
            ServerConstants.get_mlops_url())
        return run_ops_url

    @staticmethod
    def get_run_stop_url():
        run_ops_url = "{}/fedmlOpsServer/api/v1/application/stopApplicationFromCli".format(
            ServerConstants.get_mlops_url())
        return run_ops_url

    @staticmethod
    def get_run_logs_url():
        run_ops_url = "{}/fedmlOpsServer/api/v1/log/getLogsFromCli".format(
            ServerConstants.get_mlops_url())
        return run_ops_url

    @staticmethod
    def get_cluster_list_url():
        cluster_list_url = "{}/fedmlOpsServer/api/v1/cli/getClusterStatus".format(
            ServerConstants.get_mlops_url())
        return cluster_list_url

    @staticmethod
    def get_cluster_start_url():
        cluster_start_url = "{}/fedmlOpsServer/api/v1/cli/startCluster".format(
            ServerConstants.get_mlops_url())
        return cluster_start_url

    @staticmethod
    def get_cluster_stop_url():
        cluster_stop_url = "{}/fedmlOpsServer/api/v1/cli/stopCluster".format(
            ServerConstants.get_mlops_url())
        return cluster_stop_url

    @staticmethod
    def get_cluster_autostop_url():
        cluster_autostop_url = "{}/fedmlOpsServer/api/v1/cli/setAutomaticStopTime".format(
            ServerConstants.get_mlops_url())
        return cluster_autostop_url

    @staticmethod
    def get_cluster_kill_url():
        cluster_stop_url = "{}/fedmlOpsServer/api/v1/cli/shutDownCluster".format(
            ServerConstants.get_mlops_url())
        return cluster_stop_url

    @staticmethod
    def get_cluster_confirm_url():
        cluster_confirm_url = "{}/fedmlOpsServer/api/v1/cli/confirmClusterMachines".format(
            ServerConstants.get_mlops_url())
        return cluster_confirm_url

    @staticmethod
    def get_app_create_url():
        app_url = "{}/fedmlOpsServer/api/v1/application/createApplicationFromCli".format(
            ServerConstants.get_mlops_url())
        return app_url

    @staticmethod
    def get_app_update_url():
        app_url = "{}/fedmlOpsServer/api/v1/application/updateApplicationFromCli".format(
            ServerConstants.get_mlops_url())
        return app_url

    @staticmethod
    def get_app_update_with_app_id_url():
        app_url = "{}/fedmlOpsServer/api/v1/application/updateApplicationConfigFromCli".format(
            ServerConstants.get_mlops_url())
        return app_url

    @staticmethod
    def get_heartbeat_url():
        heartbeat_url = "{}/fedmlOpsServer/api/v1/cli/heartBeat".format(
            ServerConstants.get_mlops_url())
        return heartbeat_url

    @staticmethod
    def get_resource_url():
        resource_url = "{}/fedmlOpsServer/api/v1/cli/resourceType".format(
            ServerConstants.get_mlops_url())
        print("resource_url: ", resource_url)
        return resource_url

    @staticmethod
    def get_user_url():
        user_url = "{}/fedmlOpsServer/api/v1/cli/getUser".format(
            ServerConstants.get_mlops_url())
        return user_url

    @staticmethod
    def get_dataset_url():
        create_dataset_url = "{}/fedmlOpsServer/api/v1/cli/dataset".format(
            ServerConstants.get_mlops_url())
        return create_dataset_url

    @staticmethod
    def get_presigned_multi_part_url():
        get_presigned_multi_part_url = "{}/system/api/v1/cli/oss/multipart/presigned-url".format(
            ServerConstants.get_mlops_url()
        )
        return get_presigned_multi_part_url

    @staticmethod
    def get_complete_multipart_upload_url():
        complete_multipart_upload_url = "{}/system/api/v1/cli/oss/multipart/upload/complete".format(
            ServerConstants.get_mlops_url()
        )
        return complete_multipart_upload_url

    @staticmethod
    def list_dataset_url():
        list_dataset_url = "{}/fedmlOpsServer/api/v1/cli/dataset/list".format(
            ServerConstants.get_mlops_url())
        return list_dataset_url

    @staticmethod
    def get_dataset_metadata_url():
        get_dataset_metadata_url = "{}/fedmlOpsServer/api/v1/cli/dataset/meta".format(
            ServerConstants.get_mlops_url())
        return get_dataset_metadata_url

    @staticmethod
    def cleanup_run_process(run_id, not_kill_subprocess=False):
        RunProcessUtils.cleanup_run_process(
            run_id, ServerConstants.get_data_dir(), ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME,
            not_kill_subprocess=not_kill_subprocess)

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
    def get_learning_process_list(run_id):
        return RunProcessUtils.get_run_process_list(
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
    def log_callback(data, is_err=True, process_obj=None, error_processor=None, should_write_log=True):
        if not is_err:
            if should_write_log:
                for data_line in data:
                    logging.info(data_line)
        else:
            error_list = list()
            for data_line in data:
                if process_obj.returncode is None or process_obj.returncode == 0:
                    if should_write_log:
                        logging.info(data_line)
                else:
                    if should_write_log:
                        logging.error(data_line)
                    error_list.append(data_line)

            if error_processor is not None and len(error_list) > 0:
                error_processor(error_list)

    @staticmethod
    def execute_commands_with_live_logs(cmds, join='&&', should_write_log_file=True,
                                        callback=None, error_processor=None):
        error_list = list()
        script_process = subprocess_with_live_logs.Popen(
            join.join(cmds), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if script_process is None:
            return None, error_list

        if callback is not None:
            callback(script_process.pid)

        exec_out_str, exec_err_str, exec_out_list, exec_err_list, latest_lines_err_list = None, None, None, None, None
        try:
            exec_out_str, exec_err_str, exec_out_list, exec_err_list, latest_lines_err_list = \
                script_process.communicate(
                    timeout=100, data_arrived_callback=ServerConstants.log_callback,
                    error_processor=error_processor, should_write_log=should_write_log_file
            )
        except Exception as e:
            pass

        if script_process.returncode is not None and script_process.returncode != 0:
            if exec_err_str is not None:
                for err_line in latest_lines_err_list:
                    err_str = sys_utils.decode_byte_str(err_line)
                    error_list.append(err_str)

                if error_processor is not None and len(error_list) > 0:
                    error_processor(error_list)

            for error_info in error_list:
                logging.error(error_info)

        return script_process, error_list

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

    @staticmethod
    def get_fedml_parent_pid_file():
        data_dir = ServerConstants.get_data_dir()
        return os.path.join(data_dir, ServerConstants.FEDML_PARENT_PID_FILE)

    @staticmethod
    def remove_fedml_parent_pid_file():
        ppid_file = ServerConstants.get_fedml_parent_pid_file()
        try:
            if os.path.exists(ppid_file):
                os.remove(ppid_file)
        except Exception as e:
            pass

    @staticmethod
    def request(url: str, json_data: dict):
        cert_path = MLOpsConfigs.get_cert_path_with_version()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, verify=True, headers=ServerConstants.API_HEADERS, json=json_data
                )
            except requests.exceptions.SSLError as err:
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    url, verify=True, headers=ServerConstants.API_HEADERS, json=json_data
                )
        else:
            response = requests.post(
                url, verify=True, headers=ServerConstants.API_HEADERS, json=json_data
            )
        return response

