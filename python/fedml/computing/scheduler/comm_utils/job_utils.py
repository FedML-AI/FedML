import json
import logging
import os
import platform
import time
import traceback
from multiprocessing import Process

import GPUtil
import psutil
from fedml import mlops

from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.comm_utils.sys_utils import get_python_program
from fedml.computing.scheduler.scheduler_core.compute_cache_manager import ComputeCacheManager
from fedml.computing.scheduler.slave import client_data_interface
from fedml.computing.scheduler.master import server_data_interface
from fedml.computing.scheduler.model_scheduler import device_client_data_interface
from fedml.computing.scheduler.model_scheduler import device_server_data_interface
from fedml.core.common.singleton import Singleton
import threading

from ..model_scheduler.device_http_proxy_inference_protocol import FedMLHttpProxyInference
from ..model_scheduler.device_model_cache import FedMLModelCache
from ..slave import client_constants
from ..master import server_constants
from ..model_scheduler import device_client_constants
from ..model_scheduler import device_server_constants


class JobRunnerUtils(Singleton):
    def __init__(self):
        if not hasattr(self, "run_id_to_gpu_ids_map"):
            self.run_id_to_gpu_ids_map = dict()
        if not hasattr(self, "available_gpu_ids"):
            self.available_gpu_ids = None
        if not hasattr(self, "lock_available_gpu_ids"):
            self.lock_available_gpu_ids = threading.Lock()
        if not hasattr(self, "endpoint_unavailable_counter"):
            self.endpoint_unavailable_counter = dict()
        if not hasattr(self, "sync_data_proc"):
            self.sync_data_proc = None
        if not hasattr(self, "released_endpoints"):
            self.released_endpoints = dict()
        if not hasattr(self, "reported_runs"):
            self.reported_runs = dict()
        if not hasattr(self, "reported_runs_on_edges"):
            self.reported_runs_on_edges = dict()

    @staticmethod
    def get_instance():
        return JobRunnerUtils()

    def occupy_gpu_ids(self, run_id, request_gpu_num, device_id, inner_id=None,
                       model_master_device_id=None, model_slave_device_id=None):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            run_id = inner_id if inner_id is not None else run_id
            switchable_device_id = model_slave_device_id \
                if inner_id is not None and model_slave_device_id is not None else device_id
            with ComputeCacheManager.get_instance().get_redis_connection().lock(
                ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_lock_key(switchable_device_id, run_id)
            ):
                available_gpu_ids = self.get_available_gpu_id_list(device_id)

                available_gpu_ids = JobRunnerUtils.search_and_refresh_available_gpu_ids(available_gpu_ids)

                cuda_visible_gpu_ids_str, matched_gpu_num, _ = JobRunnerUtils.request_gpu_ids(
                    request_gpu_num, available_gpu_ids)
                if cuda_visible_gpu_ids_str is None:
                    return None

                run_gpu_ids = available_gpu_ids[0:matched_gpu_num].copy()
                available_gpu_ids = available_gpu_ids[matched_gpu_num:].copy()
                available_gpu_ids = list(dict.fromkeys(available_gpu_ids))

                with ComputeCacheManager.get_instance().get_redis_connection().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_device_lock_key(device_id)
                ):
                    ComputeCacheManager.get_instance().get_gpu_cache().set_device_available_gpu_ids(device_id, available_gpu_ids)

                ComputeCacheManager.get_instance().get_gpu_cache().set_device_run_num_gpus(switchable_device_id, run_id, matched_gpu_num)
                ComputeCacheManager.get_instance().get_gpu_cache().set_device_run_gpu_ids(switchable_device_id, run_id, run_gpu_ids)

                if model_master_device_id is not None and model_slave_device_id is not None:
                    ComputeCacheManager.get_instance().get_gpu_cache().set_edge_model_id_map(
                        run_id, device_id, model_master_device_id, model_slave_device_id)

                return cuda_visible_gpu_ids_str
        except Exception as e:
            logging.info(f"Exception {traceback.format_exc()}")
            return None

    @staticmethod
    def search_and_refresh_available_gpu_ids(available_gpu_ids):
        trimmed_gpu_ids = JobRunnerUtils.trim_unavailable_gpu_ids(available_gpu_ids)
        # if len(trimmed_gpu_ids) <= 0:
        #     available_gpu_ids = JobRunnerUtils.balance_available_gpu_ids(trimmed_gpu_ids)
        return trimmed_gpu_ids

    @staticmethod
    def balance_available_gpu_ids(available_gpu_ids):
        gpu_list, realtime_available_gpu_ids = JobRunnerUtils.get_gpu_list_and_realtime_gpu_available_ids()
        available_gpu_ids = realtime_available_gpu_ids
        if len(available_gpu_ids) <= 0:
            for gpu in gpu_list:
                gpu = GPUtil.GPU(gpu)
                if gpu.memoryUtil > 0.8:
                    continue
                available_gpu_ids.append(gpu.id)

        return available_gpu_ids.copy()

    @staticmethod
    def request_gpu_ids(request_gpu_num, available_gpu_ids):
        available_gpu_count = len(available_gpu_ids)
        request_gpu_num = 0 if request_gpu_num is None else request_gpu_num
        matched_gpu_num = min(available_gpu_count, request_gpu_num)
        if matched_gpu_num <= 0 or matched_gpu_num != request_gpu_num:
            return None, None, None

        matched_gpu_ids = map(lambda x: str(x), available_gpu_ids[0:matched_gpu_num])
        cuda_visible_gpu_ids_str = ",".join(matched_gpu_ids)
        return cuda_visible_gpu_ids_str, matched_gpu_num, matched_gpu_ids

    @staticmethod
    def trim_unavailable_gpu_ids(gpu_ids):
        # Trim the gpu ids based on the realtime available gpu id list.
        gpu_list, realtime_available_gpu_ids = JobRunnerUtils.get_gpu_list_and_realtime_gpu_available_ids()
        unavailable_gpu_ids = list()
        for index, gpu_id in enumerate(gpu_ids):
            if int(gpu_id) not in realtime_available_gpu_ids:
                unavailable_gpu_ids.append(index)

        trimmed_gpu_ids = [gpu_id for index, gpu_id in enumerate(gpu_ids) if index not in unavailable_gpu_ids]

        return trimmed_gpu_ids.copy()

    def release_gpu_ids(self, run_id, device_id):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            with ComputeCacheManager.get_instance().get_redis_connection().lock(
                ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_lock_key(device_id, run_id)
            ):
                run_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_gpu_ids(device_id, run_id)
                if run_gpu_ids is None:
                    return

                edge_device_id, model_master_device_id, model_slave_device_id = \
                    ComputeCacheManager.get_instance().get_gpu_cache().get_edge_model_id_map(run_id)
                if edge_device_id is None:
                    edge_device_id = device_id

                available_gpu_ids = self.get_available_gpu_id_list(edge_device_id)

                available_gpu_ids.extend(run_gpu_ids.copy())
                available_gpu_ids = list(dict.fromkeys(available_gpu_ids))

                with ComputeCacheManager.get_instance().get_redis_connection().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_device_lock_key(device_id)
                ):
                    ComputeCacheManager.get_instance().get_gpu_cache().set_device_available_gpu_ids(edge_device_id, available_gpu_ids)

                ComputeCacheManager.get_instance().get_gpu_cache().set_device_run_gpu_ids(device_id, run_id, None)
        except Exception as e:
            logging.info(f"Exception {traceback.format_exc()}")
            pass

    def sync_data_on_startup(self, edge_id):
        if self.sync_data_proc is None:
            self.sync_data_proc = Process(target=JobRunnerUtils.sync_proc, args=(edge_id,))
            self.sync_data_proc.start()

    @staticmethod
    def sync_proc(edge_id):
        JobRunnerUtils.get_instance().reset_available_gpu_id_list(edge_id)
        JobRunnerUtils.get_instance().sync_run_process_gpu()
        JobRunnerUtils.get_instance().sync_endpoint_process_gpu()

    def sync_run_process_gpu(self):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            with ComputeCacheManager.get_instance().get_redis_connection().lock(
                ComputeCacheManager.get_instance().get_gpu_cache().get_run_info_sync_lock_key("")
            ):
                count = 0
                job_list = client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()
                for job in job_list.job_list:
                    count += 1
                    if count >= 1000:
                        break

                    run_process_list = client_constants.ClientConstants.get_learning_process_list(job.job_id)
                    all_run_processes_exited = True if len(run_process_list) <= 0 else False

                    if SchedulerConstants.is_run_completed(job.status):
                        client_constants.ClientConstants.cleanup_learning_process(job.job_id)
                        all_run_processes_exited = True

                    if all_run_processes_exited:
                        self.release_gpu_ids(job.job_id, job.edge_id)
        except Exception as e:
            logging.info(f"Exception when syncing run process.{traceback.format_exc()}")
            pass

    def sync_endpoint_process_gpu(self):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            with ComputeCacheManager.get_instance().get_redis_connection().lock(
                ComputeCacheManager.get_instance().get_gpu_cache().get_run_info_sync_lock_key("")
            ):
                count = 0
                FedMLModelCache.get_instance().set_redis_params()
                device_client_data_interface.FedMLClientDataInterface.get_instance().create_job_table()
                job_list = device_client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()
                for job in job_list.job_list:
                    count += 1
                    if count >= 1000:
                        break
                    if job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                        job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED:
                        self.release_gpu_ids(job.job_id, job.edge_id)
        except Exception as e:
            logging.info("Exception when syncing endpoint process.")
            pass

    def get_available_gpu_id_list(self, device_id):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            with ComputeCacheManager.get_instance().get_redis_connection().lock(
                ComputeCacheManager.get_instance().get_gpu_cache().get_device_lock_key(device_id)
            ):
                available_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_available_gpu_ids(device_id)
                if available_gpu_ids is None:
                    gpu_ids = JobRunnerUtils.get_realtime_gpu_available_ids().copy()
                    ComputeCacheManager.get_instance().get_gpu_cache().set_device_available_gpu_ids(device_id, gpu_ids)
                    available_gpu_ids = gpu_ids

                self.available_gpu_ids = available_gpu_ids
                return self.available_gpu_ids
        except Exception as e:
            logging.info(f"Exception {traceback.format_exc()}")
            return []

    @staticmethod
    def reset_available_gpu_id_list(device_id):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            with ComputeCacheManager.get_instance().get_redis_connection().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_device_lock_key(device_id)
            ):
                current_available_gpu_ids = JobRunnerUtils.get_realtime_gpu_available_ids().copy()
                ComputeCacheManager.get_instance().get_gpu_cache().set_device_available_gpu_ids(device_id, current_available_gpu_ids)
        except Exception as e:
            logging.info(f"Exception {traceback.format_exc()}")
            pass

    @staticmethod
    def get_realtime_gpu_available_ids():
        gpu_list = sys_utils.get_gpu_list()
        gpu_count = len(gpu_list)
        realtime_available_gpu_ids = sys_utils.get_available_gpu_id_list(limit=gpu_count)
        return realtime_available_gpu_ids

    @staticmethod
    def get_gpu_list_and_realtime_gpu_available_ids():
        gpu_list = sys_utils.get_gpu_list()
        gpu_count = len(gpu_list)
        realtime_available_gpu_ids = sys_utils.get_available_gpu_id_list(limit=gpu_count)
        return gpu_list, realtime_available_gpu_ids

    @staticmethod
    def generate_job_execute_commands(run_id, edge_id, version,
                                      package_type, executable_interpreter, entry_file_full_path,
                                      conf_file_object, entry_args, assigned_gpu_ids,
                                      job_api_key, client_rank, job_yaml=None, request_gpu_num=None,
                                      scheduler_match_info=None, cuda_visible_gpu_ids_str=None):
        shell_cmd_list = list()
        entry_commands_origin = list()
        computing = job_yaml.get("computing", {})
        request_gpu_num = computing.get("minimum_num_gpus", None) if request_gpu_num is None else request_gpu_num

        # Read entry commands if job is from launch
        if package_type == SchedulerConstants.JOB_PACKAGE_TYPE_LAUNCH or \
                os.path.basename(entry_file_full_path) == SchedulerConstants.LAUNCH_JOB_DEFAULT_ENTRY_NAME:
            with open(entry_file_full_path, 'r') as entry_file_handle:
                entry_commands_origin.extend(entry_file_handle.readlines())
                entry_file_handle.close()

        # Generate the export env list for publishing environment variables
        export_cmd = "set" if platform.system() == "Windows" else "export"
        export_config_env_list, config_env_name_value_map = JobRunnerUtils.parse_config_args_as_env_variables(
            export_cmd, conf_file_object)

        # Generate the export env list about scheduler matching info for publishing environment variables
        export_match_env_list, match_env_name_value_map = \
            JobRunnerUtils.assign_matched_resources_to_run_and_generate_envs(
                run_id, export_cmd, scheduler_match_info
            )

        # Replace entry commands with environment variable values
        entry_commands = JobRunnerUtils.replace_entry_command_with_env_variable(
            entry_commands_origin, config_env_name_value_map
        )
        entry_commands = JobRunnerUtils.replace_entry_command_with_env_variable(
            entry_commands, match_env_name_value_map
        )

        # Replace entry arguments with environment variable values
        entry_args = JobRunnerUtils.replace_entry_args_with_env_variable(entry_args, config_env_name_value_map)
        entry_args = JobRunnerUtils.replace_entry_args_with_env_variable(entry_args, match_env_name_value_map)

        # Add the export env list to the entry commands
        for config_env_cmd in export_config_env_list:
            entry_commands.insert(0, config_env_cmd)
        for match_env_cmd in export_match_env_list:
            entry_commands.insert(0, match_env_cmd)

        # Add general environment variables
        entry_commands.insert(0, f"{export_cmd} FEDML_CURRENT_EDGE_ID={edge_id}\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_CURRENT_RUN_ID={run_id}\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_CURRENT_VERSION={version}\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_ENV_VERSION={version}\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_USING_MLOPS=true\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_CLIENT_RANK={client_rank}\n")
        if job_api_key is not None and str(job_api_key).strip() != "":
            random_out = sys_utils.random2(job_api_key, "FEDML@88119999GREAT")
            random_list = random_out.split("FEDML_NEXUS@")
            entry_commands.insert(0, f"{export_cmd} FEDML_RUN_API_KEY={random_list[1]}\n")
        if cuda_visible_gpu_ids_str is not None and str(cuda_visible_gpu_ids_str).strip() != "":
            entry_commands.insert(0, f"{export_cmd} CUDA_VISIBLE_DEVICES={cuda_visible_gpu_ids_str}\n")
        print(f"cuda_visible_gpu_ids_str {cuda_visible_gpu_ids_str}")

        # Set -e for the entry script
        entry_commands_filled = list()
        if platform.system() == "Windows":
            entry_file_full_path = entry_file_full_path.rstrip(".sh") + ".bat"
            for cmd in entry_commands:
                entry_commands_filled.append(cmd)
                entry_commands_filled.append("if %ERRORLEVEL% neq 0 EXIT %ERRORLEVEL%\n")
            entry_commands_filled.append("EXIT %ERRORLEVEL%")
        else:
            entry_commands_filled = entry_commands
            entry_commands_filled.insert(0, "set -e\n")

        # If the job type is not launch, we need to generate an entry script wrapping with entry commands
        if package_type != SchedulerConstants.JOB_PACKAGE_TYPE_LAUNCH and \
                os.path.basename(entry_file_full_path) != SchedulerConstants.LAUNCH_JOB_DEFAULT_ENTRY_NAME:
            if str(entry_file_full_path).endswith(".sh"):
                shell_program = SchedulerConstants.CLIENT_SHELL_BASH
            elif str(entry_file_full_path).endswith(".py"):
                shell_program = get_python_program()
            elif str(entry_file_full_path).endswith(".bat"):
                shell_program = SchedulerConstants.CLIENT_SHELL_PS
            entry_commands_filled.append(f"{shell_program} {entry_file_full_path} {entry_args}\n")
            entry_file_full_path = os.path.join(
                os.path.dirname(entry_file_full_path), os.path.basename(entry_file_full_path) + ".sh")

        # Write the entry commands to the entry script
        with open(entry_file_full_path, 'w') as entry_file_handle:
            entry_file_handle.writelines(entry_commands_filled)
            entry_file_handle.close()

        # Generate the shell commands to be executed
        shell_cmd_list.append(f"{executable_interpreter} {entry_file_full_path}")

        return shell_cmd_list

    @staticmethod
    def replace_entry_command_with_env_variable(entry_commands, env_name_value_map):
        entry_commands_replaced = list()
        for entry_cmd in entry_commands:
            for env_name, env_value in env_name_value_map.items():
                if platform.system() == "Windows":
                    entry_cmd = entry_cmd.replace(f"%{env_name}%", str(env_value))
                else:
                    entry_cmd = entry_cmd.replace(f"${{{env_name}}}", str(env_value))
                    entry_cmd = entry_cmd.replace(f"${env_name}", str(env_value))

            entry_commands_replaced.append(entry_cmd)

        return entry_commands_replaced

    @staticmethod
    def replace_entry_args_with_env_variable(entry_args, env_name_value_map):
        if entry_args is None:
            return ""
        for env_name, env_value in env_name_value_map.items():
            if platform.system() == "Windows":
                entry_args = entry_args.replace(f"%{env_name}%", str(env_value))
            else:
                entry_args = entry_args.replace(f"${{{env_name}}}", str(env_value))
                entry_args = entry_args.replace(f"${env_name}", str(env_value))

        return entry_args

    @staticmethod
    def parse_config_args_as_env_variables(export_cmd, run_params):
        export_env_command_list, env_name_value_map = JobRunnerUtils.get_env_from_dict(
            export_cmd, run_params
        )

        return export_env_command_list, env_name_value_map

    @staticmethod
    def get_env_from_dict(
            export_cmd, config_dict, export_env_command_list=[], env_name_value_map=dict(),
            config_key_path=""
    ):
        if config_dict == {}:
            return {}

        for config_key, config_value in config_dict.items():
            if isinstance(config_value, dict):
                JobRunnerUtils.get_env_from_dict(
                    export_cmd, config_value, export_env_command_list=export_env_command_list,
                    env_name_value_map=env_name_value_map, config_key_path=config_key
                )
            else:
                env_name = f"FEDML_ENV_{'' if config_key_path == '' else str(config_key_path).upper() + '_'}" \
                           f"{str(config_key).upper()}"
                config_value = str(config_value).replace("\n", ";")
                config_value = str(config_value).replace("\"", "\\\"")
                export_env_command_list.append(f"{export_cmd} {env_name}=\"{config_value}\"\n")
                env_name_value_map[env_name] = config_value

        return export_env_command_list, env_name_value_map

    @staticmethod
    def assign_matched_resources_to_run_and_generate_envs(run_id, export_cmd, scheduler_match_info):
        if scheduler_match_info is None:
            scheduler_match_info = {}
        master_node_addr = scheduler_match_info.get("master_node_addr", "localhost")
        master_node_port = scheduler_match_info.get(
            "master_node_port", SchedulerConstants.JOB_MATCH_DEFAULT_MASTER_NODE_PORT)
        num_nodes = scheduler_match_info.get("num_nodes", 1)
        matched_gpu_num = scheduler_match_info.get("matched_gpu_num", 0)
        matched_gpu_ids = scheduler_match_info.get("matched_gpu_ids", None)
        matched_gpu_num = 1 if matched_gpu_num <= 0 else matched_gpu_num

        export_env_command_list = list()
        env_name_value_map = dict()

        if master_node_addr is not None and str(master_node_addr).strip() != "":
            export_env_command_list.append(f"{export_cmd} FEDML_NODE_0_ADDR={master_node_addr}\n")
            env_name_value_map["FEDML_NODE_0_ADDR"] = master_node_addr

        if master_node_port is not None and str(master_node_port).strip() != "":
            export_env_command_list.append(f"{export_cmd} FEDML_NODE_0_PORT={master_node_port}\n")
            env_name_value_map["FEDML_NODE_0_PORT"] = master_node_port

        if num_nodes is not None and str(num_nodes).strip() != "":
            export_env_command_list.append(f"{export_cmd} FEDML_NUM_NODES={num_nodes}\n")
            env_name_value_map["FEDML_NUM_NODES"] = num_nodes

        return export_env_command_list, env_name_value_map

    def monitor_slave_run_process_status(self):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            count = 0
            job_list = client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()
            for job in job_list.job_list:
                count += 1
                if count >= 1000:
                    break

                # Calc the timeout
                started_time = int(float(job.started_time))
                timeout = time.time() - started_time

                # Check if all processes of the specific run are exited
                run_process_list = client_constants.ClientConstants.get_learning_process_list(job.job_id)
                all_run_processes_exited = True if len(run_process_list) <= 0 else False

                # Get the timeout threshold
                timeout_threshold = None
                if job.status == client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_PROVISIONING or \
                        job.status == client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_QUEUED:
                    timeout_threshold = SchedulerConstants.TRAIN_PROVISIONING_TIMEOUT
                elif job.status == client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_INITIALIZING or \
                        job.status == client_constants.ClientConstants.MSG_MLOPS_RUN_STATUS_STARTING or \
                        job.status == client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_UPGRADING:
                    timeout_threshold = SchedulerConstants.TRAIN_STARTING_TIMEOUT
                elif job.status == client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING or \
                        job.status == client_constants.ClientConstants.MSG_MLOPS_RUN_STATUS_RUNNING:
                    timeout_threshold = SchedulerConstants.TRAIN_RUNNING_TIMEOUT
                elif job.status == client_constants.ClientConstants.MSG_MLOPS_RUN_STATUS_STOPPING:
                    timeout_threshold = SchedulerConstants.TRAIN_STOPPING_TIMEOUT

                # If the run processes have exited but run status is not completed and
                # timeout is out of the range, then release gpu ids and report failed status to the master agent.
                if all_run_processes_exited and not SchedulerConstants.is_run_completed(job.status) and \
                        timeout_threshold is not None and timeout > timeout_threshold:
                    # Release the gpu ids
                    self.release_gpu_ids(job.job_id, job.edge_id)

                    # Report failed status to the master agent
                    mlops.log_training_failed_status(
                        run_id=job.job_id, edge_id=job.edge_id, enable_broadcast=True)

                    print(f"[Slave][{job.job_id}:{job.edge_id}] Due to timeout, release gpu ids and "
                          f"set run status of slave to failed.")

        except Exception as e:
            logging.info(f"Exception when monitoring run process on the slave agent.{traceback.format_exc()}")
            pass

    def monitor_master_run_process_status(self, server_id, device_info_reporter=None):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            count = 0
            job_list = server_data_interface.FedMLServerDataInterface.get_instance().get_jobs_from_db()
            for job in job_list.job_list:
                count += 1
                if count >= 1000:
                    break

                # Calc the timeout
                started_time = int(float(job.started_time))
                timeout = time.time() - started_time

                # Get the timeout threshold
                timeout_threshold = None
                if job.status == server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_PROVISIONING:
                    timeout_threshold = SchedulerConstants.TRAIN_PROVISIONING_TIMEOUT
                elif job.status == server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_STARTING or \
                        job.status == server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_UPGRADING:
                    timeout_threshold = SchedulerConstants.TRAIN_STARTING_TIMEOUT
                elif job.status == server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING:
                    timeout_threshold = SchedulerConstants.TRAIN_RUNNING_TIMEOUT
                elif job.status == server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_STOPPING:
                    timeout_threshold = SchedulerConstants.TRAIN_STOPPING_TIMEOUT

                # Check if the timout is greater than the threshold value
                if timeout_threshold is not None and timeout > timeout_threshold:
                    # Report failed status to the master agent
                    mlops.log_aggregation_failed_status(run_id=job.job_id, edge_id=server_id)
                    print(f"[Master][{job.job_id}:{job.edge_id}:{server_id}] Due to timeout, set run status to failed.")

                # Request all running process list from the edge device.
                if not SchedulerConstants.is_run_completed(job.status) and \
                        timeout > SchedulerConstants.TRAIN_INIT_TIMEOUT:
                    run_completed_on_all_edges = True
                    job_run_json = json.loads(job.running_json)
                    edge_ids = job_run_json.get("edgeids", [])
                    for edge_id in edge_ids:
                        device_info = device_info_reporter.get_device_info(job.job_id, edge_id, server_id)
                        if device_info is None:
                            continue
                        run_process_list_map = device_info.get("run_process_list_map", {})
                        run_process_list = run_process_list_map.get(str(job.job_id), [])
                        if len(run_process_list) <= 0:
                            # Report failed status to the master agent
                            if self.reported_runs_on_edges.get(str(job.job_id)) is None:
                                self.reported_runs_on_edges[str(job.job_id)] = dict()
                            if not self.reported_runs_on_edges[str(job.job_id)].get(str(edge_id), False):
                                self.reported_runs_on_edges[str(job.job_id)][str(edge_id)] = True

                                mlops.log_training_failed_status(run_id=job.job_id, edge_id=edge_id)
                                mlops.log_run_log_lines(
                                    job.job_id, edge_id, ["ERROR: Client process exited------------------------------"],
                                    SchedulerConstants.JOB_TASK_TYPE_TRAIN)
                                print(f"[Master][{job.job_id}:{edge_id}] Due to job terminated on the slave agent, "
                                      f"set run status of slave to failed.")
                        else:
                            run_completed_on_all_edges = False

                    # If run completed on all edges, then report run status to the master agent.
                    if run_completed_on_all_edges:
                        # Report failed status to the master agent
                        if not self.reported_runs.get(str(job.job_id), False):
                            self.reported_runs[str(job.job_id)] = True

                            mlops.log_run_log_lines(
                                job.job_id, job.edge_id, ["ERROR: Run failed ------------------------------"],
                                SchedulerConstants.JOB_TASK_TYPE_TRAIN)
                            mlops.log_aggregation_failed_status(run_id=job.job_id, edge_id=server_id)
                            print(f"[Master][{job.job_id}:{job.edge_id}:{server_id}] "
                                  f"Due to job failed on all slave agents, set run status to failed.")

        except Exception as e:
            logging.info(f"Exception when monitoring run process on the master agent.{traceback.format_exc()}")
            pass

    def monitor_slave_endpoint_status(self):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            count = 0
            FedMLModelCache.get_instance().set_redis_params()
            device_client_data_interface.FedMLClientDataInterface.get_instance().create_job_table()
            job_list = device_client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()
            for job in job_list.job_list:
                count += 1
                if count >= 1000:
                    break

                endpoint_status = FedMLModelCache.get_instance().get_end_point_status(job.job_id)
                if job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED:
                    if not self.released_endpoints.get(str(job.job_id), False):
                        self.released_endpoints[str(job.job_id)] = True

                        # Release the gpu ids
                        self.release_gpu_ids(job.job_id, job.edge_id)
                        print(f"[Worker][{job.job_id}:{job.edge_id}] Release gpu ids.")

                elif job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED:
                    endpoint_json = json.loads(job.running_json)
                    model_config = endpoint_json.get("model_config", {})
                    model_name = model_config.get("model_name", None)
                    endpoint_name = endpoint_json.get("end_point_name", None)

                    if model_name is not None and endpoint_status is not None and \
                            endpoint_status != device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE:
                        # Get model deployment result
                        deployment_result, activated = \
                            FedMLModelCache.get_instance().get_deployment_result_with_device_id(
                                job.job_id, endpoint_name, model_name, job.edge_id)
                        if not activated:
                            print(f"[Worker][{job.job_id}:{job.edge_id}] Due to not activated, set endpoint status "
                                  f"to offline at the status {job.status}.")
                            mlops.log_endpoint_status(
                                job.job_id,
                                device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE)
                            FedMLModelCache.get_instance().set_end_point_status(
                                job.job_id, endpoint_name,
                                device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE)
                            continue

                        # Check the endpoint status
                        self._check_slave_endpoint_status(job.job_id, job.edge_id, deployment_result)

        except Exception as e:
            logging.info(f"[Worker] Exception when syncing endpoint process on the slave agent. {traceback.format_exc()}")
            pass

    def _check_slave_endpoint_status(self, endpoint_id, device_id, deployment_result):
        result_json = deployment_result
        inference_url = result_json.get("model_url", None)
        model_metadata = result_json.get("model_metadata", {})
        input_list = model_metadata.get("inputs", None)
        output_list = []

        # Run inference request to check if endpoint is running normally.
        while True:
            response_ok, inference_response = \
                FedMLHttpProxyInference.run_http_proxy_inference_with_request(
                    endpoint_id, inference_url, input_list, output_list,
                    timeout=SchedulerConstants.ENDPOINT_STATUS_CHECK_TIMEOUT)
            if self.endpoint_unavailable_counter.get(str(endpoint_id)) is None:
                self.endpoint_unavailable_counter[str(endpoint_id)] = 0
            if not response_ok:
                self.endpoint_unavailable_counter[str(endpoint_id)] += 1
            else:
                self.endpoint_unavailable_counter[str(endpoint_id)] = 0
                return True

            # If the endpoint unavailable counter is greater than the threshold value,
            # then release gpu ids and report failed status to the master agent.
            if self.endpoint_unavailable_counter.get(str(endpoint_id), 0) > \
                    SchedulerConstants.ENDPOINT_FAIL_THRESHOLD_VALUE:
                if not self.released_endpoints.get(str(endpoint_id), False):
                    self.released_endpoints[str(endpoint_id)] = True

                    # Release the gpu ids
                    self.release_gpu_ids(endpoint_id, device_id)

                    # Report failed status to the master agent
                    print(f"[{endpoint_id}:{device_id}] Set worker status to offline due to unable to get response.")
                    mlops.log_training_status(
                        device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_OFFLINE,
                        run_id=endpoint_id, edge_id=device_id, is_from_model=True, enable_broadcast=True)
                return False
            time.sleep(2)

    def monitor_master_endpoint_status(self):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            count = 0
            FedMLModelCache.get_instance().set_redis_params()
            device_server_data_interface.FedMLServerDataInterface.get_instance().create_job_table()
            job_list = device_server_data_interface.FedMLServerDataInterface.get_instance().get_jobs_from_db()
            for job in job_list.job_list:
                count += 1
                if count >= 1000:
                    break

                endpoint_status = FedMLModelCache.get_instance().get_end_point_status(job.job_id)
                endpoint_json = json.loads(job.running_json) if job.running_json is not None else {}
                model_config = endpoint_json.get("model_config", {})
                model_name = model_config.get("model_name", None)
                endpoint_name = endpoint_json.get("end_point_name", None)

                if endpoint_status == device_server_constants.ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED:
                    # Release the gpu ids
                    self.release_gpu_ids(job.job_id, job.edge_id)
                elif endpoint_status == device_server_constants.ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
                    if model_name is not None:
                        try:
                            # Get model deployment result
                            is_endpoint_offline = True
                            result_list = FedMLModelCache.get_instance().get_deployment_result_list(
                                job.job_id, endpoint_name, model_name)
                            for result_item in result_list:
                                result_device_id, result_payload = FedMLModelCache.get_instance().get_result_item_info(result_item)

                                # Check if the endpoint is activated
                                end_point_activated = FedMLModelCache.get_instance().get_end_point_activation(job.job_id)
                                if not end_point_activated:
                                    print(f"[Master][{job.job_id}] Endpoint is not activated, set status to "
                                          f"offline after deployed")
                                    mlops.log_endpoint_status(
                                        job.job_id,
                                        device_server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE)
                                    FedMLModelCache.get_instance().set_end_point_status(
                                        job.job_id, endpoint_name,
                                        device_server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE)
                                    break
                                else:
                                    # Check if the endpoint is running
                                    if self._check_slave_endpoint_status(job.job_id, result_device_id, result_payload):
                                        is_endpoint_offline = False

                            # If the endpoint is offline, then report offline status to the MLOps.
                            if is_endpoint_offline:
                                print(f"[Master][{job.job_id}] Due to all worker is offline, set endpoint status to "
                                      f"offline after deployed .")
                                mlops.log_endpoint_status(
                                    job.job_id,
                                    device_server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE)
                                FedMLModelCache.get_instance().set_end_point_status(
                                    job.job_id, endpoint_name,
                                    device_server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE)

                        except Exception as e:
                            print(f"[Master][{job.job_id}] Exception when check endpoint status: "
                                  f"{traceback.format_exc()}")
                elif endpoint_status == \
                        device_server_constants.ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_INITIALIZING:
                    started_time = int(float(job.started_time))
                    timeout = time.time() - started_time
                    if timeout > SchedulerConstants.ENDPOINT_DEPLOYMENT_PROVISIONING_TIMEOUT:
                        print(f"[Master][{job.job_id}] Due to timeout, set endpoint status to "
                              f"offline at the status {endpoint_status}.")
                        mlops.log_endpoint_status(
                            job.job_id,
                            device_server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE)
                        FedMLModelCache.get_instance().set_end_point_status(
                            job.job_id, endpoint_name,
                            device_server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE)
                elif endpoint_status == \
                        device_server_constants.ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYING:
                    started_time = int(float(job.started_time))
                    timeout = time.time() - started_time
                    if timeout > SchedulerConstants.ENDPOINT_DEPLOYMENT_DEPLOYING_TIMEOUT:
                        print(f"[Master][{job.job_id}] Due to timeout, set endpoint status to "
                              f"offline at the status {endpoint_status}.")
                        mlops.log_endpoint_status(
                            job.job_id,
                            device_server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE)
                        FedMLModelCache.get_instance().set_end_point_status(
                            job.job_id, endpoint_name,
                            device_server_constants.ServerConstants.MSG_MLOPS_SERVER_STATUS_OFFLINE)

        except Exception as e:
            logging.info(f"Exception when syncing endpoint process on the master agent {traceback.format_exc()}.")
            pass