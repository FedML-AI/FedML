import logging
import os
import platform
import subprocess
import time
import traceback
import uuid

import requests

import fedml
from fedml.computing.scheduler.comm_utils import sys_utils, security_utils
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.comm_utils.sys_utils import get_sys_runner_info
from fedml.computing.scheduler.scheduler_core.general_constants import GeneralConstants
from fedml.core.common.singleton import Singleton
from fedml.core.mlops import MLOpsConfigs


class FedMLAccountManager(Singleton):
    LOCAL_RUNNER_INFO_DIR_NAME = 'runner_infos'
    STATUS_IDLE = "IDLE"
    ROLE_EDGE_SERVER = "edge_server"
    ROLE_CLOUD_AGENT = "cloud_agent"
    ROLE_CLOUD_SERVER = "cloud_server"
    ROLE_EDGE_DEVICE = "client"
    ROLE_GPU_PROVIDER = "gpu_supplier"
    ROLE_GPU_MASTER_SERVER = "gpu_master_server"
    ROLE_DEPLOY_MASTER_ON_PREM = "md.on_premise_device.master"
    ROLE_DEPLOY_WORKER_ON_PREM = "md.on_premise_device"

    DEVICE_ID_SUFFIX_EDGE_SERVER = ".Edge.Server"
    DEVICE_ID_SUFFIX_CLOUD_AGENT = ".Public.Cloud"
    DEVICE_ID_SUFFIX_CLOUD_SERVER = ".Public.Server"
    DEVICE_ID_SUFFIX_EDGE_DEVICE = ".Edge.Device"
    DEVICE_ID_SUFFIX_GPU_PROVIDER = ".Edge.GPU.Supplier"
    DEVICE_ID_SUFFIX_GPU_MASTER_SERVER = ".Edge.GPU.MasterServer"
    DEVICE_ID_SUFFIX_DEPLOY = "MDA"
    DEVICE_ID_SUFFIX_DEPLOY_MASTER_ON_PREM = ".OnPremise.Master.Device"
    DEVICE_ID_SUFFIX_DEPLOY_WORKER_ON_PREM = ".OnPremise.Device"

    DEVICE_ID_DOCKER_TAG = ".Docker"
    DEVICE_ID_DOCKER_HUB_TAG = ".DockerHub"

    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLAccountManager()

    def login(self, user_id, api_key="", device_id=None, os_name=None, role=None, runner_cmd=None):
        # Build the agent args
        agent_args = self.build_agent_args(
            user_id, api_key=api_key, device_id=device_id, os_name=os_name, role=role, runner_cmd=runner_cmd
        )

        # Fetch configs from the MLOps config server.
        service_config = dict()
        log_server_url = None
        config_try_count = 0
        edge_id = 0
        while config_try_count < 5:
            # noinspection PyBroadException
            try:
                mqtt_config, s3_config, mlops_config, docker_config = FedMLAccountManager.fetch_configs()
                service_config["mqtt_config"] = mqtt_config
                service_config["s3_config"] = s3_config
                service_config["ml_ops_config"] = mlops_config
                service_config["docker_config"] = docker_config
                log_server_url = mlops_config.get("LOG_SERVER_URL", None)
                break
            except Exception as e:
                print("{}\n{}".format(SchedulerConstants.ERR_MSG_BINDING_EXCEPTION_1, traceback.format_exc()))
                print(SchedulerConstants.ERR_MSG_BINDING_EXIT_RETRYING)
                config_try_count += 1
                time.sleep(3)
                continue

        # Failed to fetch the config after retrying many times.
        if config_try_count >= 5:
            print("")
            print("[5] Oops, you failed to login the FedML MLOps platform.")
            print("Please check whether your network is normal!")
            return None

        # Bind account id to FedML® Nexus AI Platform
        register_try_count = 0
        edge_id = -1
        user_name = None
        extra_url = None
        general_edge_id = None
        while register_try_count < 5:
            # noinspection PyBroadException
            try:
                edge_id, user_name, extra_url, general_edge_id = FedMLAccountManager.bind_account_and_device_id(
                    service_config["ml_ops_config"]["EDGE_BINDING_URL"], agent_args.account_id,
                    agent_args.unique_device_id, agent_args.os_name,
                    api_key=api_key, role=role
                )
                if edge_id > 0:
                    break
            except SystemExit as e:
                print("Your account does not exist. Please make sure your account correct.")
                os.system("fedml logout -s")
                return
            except Exception as e:
                print("{}\n{}".format(SchedulerConstants.ERR_MSG_BINDING_EXCEPTION_2, traceback.format_exc()))
                print(SchedulerConstants.ERR_MSG_BINDING_EXIT_RETRYING)
                register_try_count += 1
                time.sleep(3)
                continue

        # Failed to bind your account after retrying many times.
        if edge_id <= 0:
            print("")
            print("[6] Oops, you failed to login the FedML MLOps platform.")
            print("Please check whether your network is normal!")
            return None

        # Fill the bound result to agent args.
        agent_args = self.fill_argent_args(
            agent_args, log_server_url=log_server_url, server_id=edge_id,
            edge_id=edge_id, general_edge_id=general_edge_id,
            user_name=user_name, extra_url=extra_url,
            agent_config=service_config)

        return agent_args

    def build_agent_args(self, user_id, api_key=None, device_id=None, os_name=None, role=None, runner_cmd=None):
        # Generate the suffix for device based on the role
        device_id_suffix = None
        is_master = False
        is_deploy = False
        if role == FedMLAccountManager.ROLE_EDGE_SERVER:
            device_id_suffix = FedMLAccountManager.DEVICE_ID_SUFFIX_EDGE_SERVER
            is_master = True
        elif role == FedMLAccountManager.ROLE_CLOUD_AGENT:
            device_id_suffix = FedMLAccountManager.DEVICE_ID_SUFFIX_CLOUD_AGENT
            is_master = True
        elif role == FedMLAccountManager.ROLE_CLOUD_SERVER:
            device_id_suffix = ""
            is_master = True
        elif role == FedMLAccountManager.ROLE_EDGE_DEVICE:
            device_id_suffix = FedMLAccountManager.DEVICE_ID_SUFFIX_EDGE_DEVICE
        elif role == FedMLAccountManager.ROLE_GPU_PROVIDER:
            device_id_suffix = FedMLAccountManager.DEVICE_ID_SUFFIX_GPU_PROVIDER
        elif role == FedMLAccountManager.ROLE_GPU_MASTER_SERVER:
            device_id_suffix = FedMLAccountManager.DEVICE_ID_SUFFIX_GPU_MASTER_SERVER
            is_master = True
        elif role == FedMLAccountManager.ROLE_DEPLOY_MASTER_ON_PREM:
            device_id_suffix = FedMLAccountManager.DEVICE_ID_SUFFIX_DEPLOY_MASTER_ON_PREM
            is_master = True
            is_deploy = True
        elif role == FedMLAccountManager.ROLE_DEPLOY_WORKER_ON_PREM:
            device_id_suffix = FedMLAccountManager.DEVICE_ID_SUFFIX_DEPLOY_WORKER_ON_PREM
            is_deploy = True

        # Build the agent args
        version = fedml.get_env_version()
        agent_args = AgentArgs()
        agent_args.role = role
        agent_args.account_id = user_id
        agent_args.api_key = api_key
        agent_args.current_running_dir = GeneralConstants.get_deploy_fedml_home_dir(is_master=is_master) \
            if is_deploy else GeneralConstants.get_launch_fedml_home_dir(is_master=is_master)
        sys_name = platform.system()
        if sys_name == "Darwin":
            sys_name = "MacOS"
        agent_args.os_name = sys_name if os_name is None or os_name == "" else os_name
        agent_args.version = version
        agent_args.log_file_dir = GeneralConstants.get_deploy_log_file_dir(is_master=is_master) \
            if is_deploy else GeneralConstants.get_launch_log_file_dir(is_master=is_master)
        is_from_docker = False
        if device_id is not None and device_id != "0":
            agent_args.current_device_id = device_id
        else:
            data_dir = GeneralConstants.get_deploy_data_dir(is_master=is_master) \
                if is_deploy else GeneralConstants.get_launch_data_dir(is_master=is_master)
            is_gpu_provider = True if role == FedMLAccountManager.ROLE_GPU_PROVIDER else False
            agent_args.current_device_id = FedMLAccountManager.get_device_id(
                data_dir=data_dir, use_machine_id=is_gpu_provider)
        agent_args.device_id = agent_args.current_device_id
        agent_args.config_version = version
        agent_args.cloud_region = ""

        # Check if it is running in the fedml docker hub
        is_from_fedml_docker_hub = False
        dock_loc_file = GeneralConstants.get_deploy_docker_location_file(is_master=is_master) \
            if is_deploy else GeneralConstants.get_launch_docker_location_file(is_master=is_master)
        if os.path.exists(dock_loc_file):
            is_from_fedml_docker_hub = True

        # Build unique device id
        docker_tag = FedMLAccountManager.DEVICE_ID_DOCKER_TAG if is_from_docker else ""
        docker_tag = FedMLAccountManager.DEVICE_ID_DOCKER_HUB_TAG if is_from_fedml_docker_hub else docker_tag
        unique_device_id = f"{agent_args.current_device_id}@{agent_args.os_name}" \
                           f"{docker_tag}{device_id_suffix}"
        if role == FedMLAccountManager.ROLE_CLOUD_SERVER:
            unique_device_id = agent_args.current_device_id

        # Set the unique device id
        agent_args.is_from_docker = is_from_docker or is_from_fedml_docker_hub
        agent_args.unique_device_id = unique_device_id
        agent_args.runner_cmd = runner_cmd

        return agent_args

    def fill_argent_args(
            self, agent_args, log_server_url=None, server_id=None, edge_id=None,
            user_name=None, extra_url=None, general_edge_id=None, agent_config=None):
        agent_args.log_server_url = log_server_url
        agent_args.server_id = server_id
        agent_args.edge_id = edge_id
        agent_args.user_name = user_name
        agent_args.extra_url = extra_url
        agent_args.general_edge_id = general_edge_id
        agent_args.agent_config = agent_config
        return agent_args

    @staticmethod
    def write_login_failed_file(is_client=True):
        login_exit_file = os.path.join(
            GeneralConstants.get_launch_log_file_dir(is_master=not is_client), "exited.log")
        with open(login_exit_file, "w") as f:
            f.writelines(f"{os.getpid()}.")

    @staticmethod
    def get_device_id(data_dir, use_machine_id=False):
        device_file_path = os.path.join(data_dir, FedMLAccountManager.LOCAL_RUNNER_INFO_DIR_NAME)
        file_for_device_id = os.path.join(device_file_path, "devices.id")
        if not os.path.exists(device_file_path):
            os.makedirs(device_file_path, exist_ok=True)
        elif os.path.exists(file_for_device_id):
            with open(file_for_device_id, 'r', encoding='utf-8') as f:
                device_id_from_file = f.readline()
                if device_id_from_file is not None and device_id_from_file != "":
                    return device_id_from_file

        if platform.system() == "Darwin":
            cmd_get_serial_num = "system_profiler SPHardwareDataType | grep Serial | awk '{gsub(/ /,\"\")}{print}' " \
                                 "|awk -F':' '{print $2}' "
            device_id = os.popen(cmd_get_serial_num).read()
            device_id = device_id.replace('\n', '').replace(' ', '')
            if device_id is None or device_id == "":
                if not use_machine_id:
                    device_id = hex(uuid.getnode())
                else:
                    device_id = FedMLAccountManager.get_gpu_machine_id()
            else:
                device_id = "0x" + device_id
        else:
            if "nt" in os.name:

                def get_uuid():
                    guid = ""
                    try:
                        cmd = "wmic csproduct get uuid"
                        guid = str(subprocess.check_output(cmd))
                        pos1 = guid.find("\\n") + 2
                        guid = guid[pos1:-15]
                    except Exception as ex:
                        logging.error(f"Failed to get uuid with Exception {ex}. Traceback: {traceback.format_exc()}")
                        pass
                    return str(guid)

                device_id = str(get_uuid())
                logging.info(device_id)
            elif "posix" in os.name:
                device_id = sys_utils.get_device_id_in_docker()
                if device_id is None:
                    if not use_machine_id:
                        device_id = hex(uuid.getnode())
                    else:
                        device_id = device_id = FedMLAccountManager.get_gpu_machine_id()
            else:
                device_id = sys_utils.run_subprocess_open(
                    "hal-get-property --udi /org/freedesktop/Hal/devices/computer --key system.hardware.uuid".split()
                )
                device_id = hex(device_id)

        if device_id is not None and device_id != "":
            with open(file_for_device_id, 'w', encoding='utf-8') as f:
                f.write(device_id)
        else:
            device_id = hex(uuid.uuid4())
            with open(file_for_device_id, 'w', encoding='utf-8') as f:
                f.write(device_id)

        return device_id

    @staticmethod
    def get_gpu_machine_id():
        gpu_list = sys_utils.get_gpu_list()
        gpu_uuids = ""
        if len(gpu_list) > 0:
            for gpu in gpu_list:
                gpu_uuids += gpu.get("uuid", "")
        else:
            gpu_uuids = str(uuid.uuid4())
        device_id_combination = \
            f"{FedMLAccountManager.get_machine_id()}-{hex(uuid.getnode())}-{gpu_uuids}"
        device_id = security_utils.get_content_hash(device_id_combination)
        return device_id

    @staticmethod
    def get_machine_id():
        try:
            import machineid
            return machineid.id().replace('\n', '').replace('\r\n', '').strip()
        except Exception as e:
            logging.error(f"Failed to get machine id with Exception {e}. Traceback: {traceback.format_exc()}")
            return hex(uuid.getnode())

    @staticmethod
    def bind_account_and_device_id(
            url, account_id, device_id, os_name, api_key="",
            role=ROLE_EDGE_SERVER):
        ip = requests.get('https://checkip.amazonaws.com').text.strip()
        fedml_ver, exec_path, os_ver, cpu_info, python_ver, torch_ver, mpi_installed, \
            cpu_usage, available_mem, total_mem, gpu_info, gpu_available_mem, gpu_total_mem, \
            gpu_count, gpu_vendor, cpu_count, gpu_device_name = get_sys_runner_info()
        host_name = sys_utils.get_host_name()
        json_params = {
            "accountid": account_id,
            "deviceid": device_id,
            "type": os_name,
            "state": FedMLAccountManager.STATUS_IDLE,
            "status": FedMLAccountManager.STATUS_IDLE,
            "processor": cpu_info,
            "core_type": cpu_info,
            "network": "",
            "role": role,
            "os_ver": os_ver,
            "memory": total_mem,
            "ip": ip,
            "api_key": api_key,
            "extra_infos": {"fedml_ver": fedml_ver, "exec_path": exec_path, "os_ver": os_ver,
                            "cpu_info": cpu_info, "python_ver": python_ver, "torch_ver": torch_ver,
                            "mpi_installed": mpi_installed, "cpu_usage": cpu_usage,
                            "available_mem": available_mem, "total_mem": total_mem,
                            "cpu_count": cpu_count, "gpu_count": 0, "host_name": host_name}
        }
        if gpu_count > 0:
            if gpu_total_mem is not None:
                json_params["gpu"] = gpu_info if gpu_info is not None else "" + ", Total GPU Memory: " + gpu_total_mem
            else:
                json_params["gpu"] = gpu_info if gpu_info is not None else ""
            json_params["extra_infos"]["gpu_info"] = gpu_info if gpu_info is not None else ""
            if gpu_available_mem is not None:
                json_params["extra_infos"]["gpu_available_mem"] = gpu_available_mem
            if gpu_total_mem is not None:
                json_params["extra_infos"]["gpu_total_mem"] = gpu_total_mem

            json_params["extra_infos"]["gpu_count"] = gpu_count
            json_params["extra_infos"]["gpu_vendor"] = gpu_vendor
            json_params["extra_infos"]["gpu_device_name"] = gpu_device_name

            gpu_available_id_list = sys_utils.get_available_gpu_id_list(limit=gpu_count)
            gpu_available_count = len(gpu_available_id_list) if gpu_available_id_list is not None else 0
            gpu_list = sys_utils.get_gpu_list()
            json_params["extra_infos"]["gpu_available_count"] = gpu_available_count
            json_params["extra_infos"]["gpu_available_id_list"] = gpu_available_id_list
            json_params["extra_infos"]["gpu_list"] = gpu_list
        else:
            json_params["gpu"] = "None"
            json_params["extra_infos"]["gpu_available_count"] = 0
            json_params["extra_infos"]["gpu_available_id_list"] = []
            json_params["extra_infos"]["gpu_list"] = []

        _, cert_path = MLOpsConfigs.get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    url, json=json_params, verify=True,
                    headers={"content-type": "application/json", "Connection": "close"}
                )
            except requests.exceptions.SSLError as err:
                logging.error(
                    f"Failed to bind account and device id with error: {err}, traceback: {traceback.format_exc()}")
                MLOpsConfigs.install_root_ca_file()
                response = requests.post(
                    url, json=json_params, verify=True,
                    headers={"content-type": "application/json", "Connection": "close"}
                )
        else:
            response = requests.post(url, json=json_params, headers={"Connection": "close"})
        edge_id, user_name, extra_url, general_edge_id = -1, None, None, None
        if response.status_code != 200:
            print(f"Binding to MLOps with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
            pass
        else:
            # print("url = {}, response = {}".format(url, response))
            status_code = response.json().get("code")
            if status_code == "SUCCESS":
                edge_id = response.json().get("data").get("id")
                user_name = response.json().get("data").get("userName", None)
                extra_url = response.json().get("data").get("url", None)
                general_edge_id = response.json().get("data").get("general_edge_id", None)
                if edge_id is None or edge_id <= 0:
                    print(f"Binding to MLOps with response.status_code = {response.status_code}, "
                          f"response.content: {response.content}")
            else:
                if status_code == SchedulerConstants.BINDING_ACCOUNT_NOT_EXIST_ERROR:
                    raise SystemExit(SchedulerConstants.BINDING_ACCOUNT_NOT_EXIST_ERROR)
                print(f"Binding to MLOps with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return -1, None, None, None
        return edge_id, user_name, extra_url, general_edge_id

    @staticmethod
    def fetch_configs():
        return MLOpsConfigs.fetch_all_configs()

    @staticmethod
    def _role_is_slave_agent(role):
        return True if role == FedMLAccountManager.ROLE_EDGE_DEVICE or \
                       role == FedMLAccountManager.ROLE_GPU_PROVIDER else False


class AgentArgs:
    def __init__(self, role=None, account_id=None, api_key=None, server_id=None, current_running_dir=None,
                 os_name=None, version=None, log_file_dir=None, log_server_url=None, device_id=None,
                 current_device_id=None, config_version=None, cloud_region=None, is_from_docker=False,
                 edge_id=None, agent_config=None, user_name=None, extra_url=None, unique_device_id=None):
        self.role = role
        self.account_id = account_id
        self.api_key = api_key
        self.current_running_dir = current_running_dir
        self.server_id = server_id
        self.os_name = os_name
        self.version = version
        self.log_file_dir = log_file_dir
        self.log_server_url = log_server_url
        self.device_id = device_id
        self.current_device_id = current_device_id
        self.config_version = config_version
        self.cloud_region = cloud_region
        self.is_from_docker = is_from_docker
        self.edge_id = edge_id
        self.client_id = edge_id
        self.agent_config = agent_config
        self.user_name = user_name
        self.extra_url = extra_url
        self.unique_device_id = unique_device_id
        self.client_id_list = None
        self.using_mlops = True
        self.server_agent_id = None
        self.general_edge_id = None
        self.runner_cmd = None

    def is_cloud_server(self):
        return self.role == FedMLAccountManager.ROLE_CLOUD_SERVER

    def is_cloud_agent(self):
        return self.role == FedMLAccountManager.ROLE_CLOUD_AGENT

    def is_edge_server(self):
        return self.role == FedMLAccountManager.ROLE_EDGE_SERVER

    def is_edge_device(self):
        return self.role == FedMLAccountManager.ROLE_EDGE_DEVICE

    def is_gpu_provider(self):
        return self.role == FedMLAccountManager.ROLE_GPU_PROVIDER

    def is_slave_agent(self):
        return self.is_edge_device() or self.is_gpu_provider()
