import logging
import os
import platform
import traceback
import docker
import fedml
from docker import errors, DockerClient
import stat

from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.computing.scheduler.comm_utils.sys_utils import get_python_program
from fedml.computing.scheduler.scheduler_core.compute_cache_manager import ComputeCacheManager
from dataclasses import dataclass, field, fields

from fedml.computing.scheduler.slave.client_data_interface import FedMLClientDataInterface
from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.comm_utils.container_utils import ContainerUtils
from typing import List
import threading
import json

run_docker_without_gpu = False


@dataclass
class DockerArgs:
    image: str = SchedulerConstants.FEDML_DEFAULT_LAUNCH_IMAGE
    username: str = ""
    password: str = ""
    registry: str = ""
    ports: List[int] = field(default_factory=lambda: [2345])


class JobRunnerUtils(Singleton):
    STATIC_RUN_LOCK_KEY_SUFFIX = "STATIC"

    def __init__(self):
        if not hasattr(self, "run_id_to_gpu_ids_map"):
            self.run_id_to_gpu_ids_map = dict()
        if not hasattr(self, "lock_available_gpu_ids"):
            self.lock_available_gpu_ids = threading.Lock()

    @staticmethod
    def get_instance():
        return JobRunnerUtils()

    def occupy_gpu_ids(self, run_id, request_gpu_num, device_id, inner_id=None,
                       model_master_device_id=None, model_slave_device_id=None):
        try:
            ComputeCacheManager.get_instance().set_redis_params()

            # For the "Switch" feature in deploy
            original_run_id = run_id
            run_id = inner_id if inner_id is not None else run_id

            # switchable_device_id is used to store the worker device id for deploy
            switchable_device_id = model_slave_device_id \
                if inner_id is not None and model_slave_device_id is not None else device_id

            logging.info(f"Request gpus on worker for run_id {run_id}: <<<<<<< "
                         f"Device id {device_id}; Switchable id {switchable_device_id}; "
                         f"Master id {model_master_device_id}; Slave id {model_slave_device_id}."
                         f" >>>>>>>")

            with ComputeCacheManager.get_instance().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_lock_key(
                        device_id, JobRunnerUtils.STATIC_RUN_LOCK_KEY_SUFFIX)
            ):
                # For launch job, the device_id plus the run_id should be a unique identifier
                run_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_gpu_ids(device_id,
                                                                                                        run_id)
                if run_gpu_ids:
                    raise Exception(f"GPUs already occupied for run_id: {run_id}, device_id: {device_id}.")

                # Incase the run id for launch job is the same as the inner id for deploy job
                if inner_id is not None and str(original_run_id) != str(inner_id):
                    ComputeCacheManager.get_instance().get_gpu_cache().set_endpoint_run_id_map(inner_id,
                                                                                               original_run_id)

                with ComputeCacheManager.get_instance().lock(
                        ComputeCacheManager.get_instance().get_gpu_cache().get_device_lock_key(device_id)
                ):

                    # Get the available GPU list, FEDML_GLOBAL_DEVICE_AVAILABLE_GPU_IDS_TAG-${device_id}
                    available_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_available_gpu_ids(
                        device_id)

                    logging.info(f"Check worker({device_id})'s realtime gpu availability in DB"
                                 f" for run {run_id}: {available_gpu_ids}")

                    # If the available GPU list is not in the cache, set it to the current system available GPU list
                    if available_gpu_ids is None:
                        # Get realtime GPU availability list from the system
                        available_gpu_ids = JobRunnerUtils.get_realtime_gpu_available_ids().copy()
                    else:
                        available_gpu_ids = JobRunnerUtils.trim_unavailable_gpu_ids(available_gpu_ids)

                    # Get the matched gpu ids string by the request gpu num
                    cuda_visible_gpu_ids_str, matched_gpu_num = JobRunnerUtils.request_gpu_ids(request_gpu_num,
                                                                                               available_gpu_ids)
                    if cuda_visible_gpu_ids_str is None:
                        if request_gpu_num:
                            error_message = (f"Failed to occupy gpu ids for run {run_id}. "
                                             f"Requested_gpu_num {request_gpu_num}; "
                                             f"Available GPU ids: {available_gpu_ids}")
                            logging.error(error_message)
                            raise Exception(error_message)
                        return None
                    else:
                        logging.info(f"Occupied GPU ids for run {run_id}: {cuda_visible_gpu_ids_str}, all"
                                     f" available GPU ids: {available_gpu_ids}")

                    # String to available set
                    run_gpu_ids = list(map(lambda x: int(x), cuda_visible_gpu_ids_str.split(",")))
                    available_gpu_ids = [gpu_id for gpu_id in available_gpu_ids if gpu_id not in set(run_gpu_ids)]
                    available_gpu_ids = list(set(available_gpu_ids))

                    ComputeCacheManager.get_instance().get_gpu_cache().set_device_available_gpu_ids(
                        device_id, available_gpu_ids)

                    # For a single run, could be scale up. So if existed such a key, should extend, not replace
                    existed_gpu_nums = ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_num_gpus(
                        switchable_device_id, run_id)
                    if existed_gpu_nums is not None and int(existed_gpu_nums) > 0:
                        matched_gpu_num += int(existed_gpu_nums)
                        device_run_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_gpu_ids(
                            switchable_device_id, run_id)
                        if run_gpu_ids is not None and device_run_gpu_ids is not None:
                            run_gpu_ids.extend(device_run_gpu_ids)
                        else:
                            logging.warning("There is inconsistency between the run_gpu_nums and the run_gpu_ids."
                                            f"existed_gpu_nums: {existed_gpu_nums}, run_gpu_ids: {run_gpu_ids}"
                                            f"device_run_gpu_ids: {device_run_gpu_ids}")

                    ComputeCacheManager.get_instance().get_gpu_cache().set_device_run_num_gpus(switchable_device_id,
                                                                                               run_id,
                                                                                               matched_gpu_num)
                    ComputeCacheManager.get_instance().get_gpu_cache().set_device_run_gpu_ids(switchable_device_id,
                                                                                              run_id,
                                                                                              run_gpu_ids)
                    ComputeCacheManager.get_instance().get_gpu_cache().set_run_device_ids(run_id,
                                                                                          [switchable_device_id])
                    ComputeCacheManager.get_instance().get_gpu_cache().set_run_total_num_gpus(run_id, matched_gpu_num)

                    if model_master_device_id is not None and model_slave_device_id is not None:
                        # Set (Launch Master, Deploy Master, Deploy Workers) device id map
                        ComputeCacheManager.get_instance().get_gpu_cache().set_edge_model_id_map(
                            run_id, device_id, model_master_device_id, model_slave_device_id)

                return cuda_visible_gpu_ids_str

        except Exception as e:
            logging.error(f"Error {e} Exception {traceback.format_exc()}")
            return None

    @staticmethod
    def search_and_refresh_available_gpu_ids(available_gpu_ids):
        trimmed_gpu_ids = JobRunnerUtils.trim_unavailable_gpu_ids(available_gpu_ids)
        return trimmed_gpu_ids

    @staticmethod
    def request_gpu_ids(request_gpu_num, available_gpu_ids):
        available_gpu_count = len(available_gpu_ids)
        request_gpu_num = 0 if request_gpu_num is None else request_gpu_num
        matched_gpu_num = min(available_gpu_count, request_gpu_num)
        if matched_gpu_num <= 0 or matched_gpu_num != request_gpu_num:
            return None, None

        matched_gpu_ids = map(lambda x: str(x), available_gpu_ids[0:matched_gpu_num])
        cuda_visible_gpu_ids_str = ",".join(matched_gpu_ids)
        return cuda_visible_gpu_ids_str, matched_gpu_num

    @staticmethod
    def trim_unavailable_gpu_ids(gpu_ids) -> List[int]:
        # Trim the gpu ids based on the realtime available gpu id list.
        available_gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
        gpu_list, realtime_available_gpu_ids = JobRunnerUtils.get_gpu_list_and_realtime_gpu_available_ids()
        unavailable_gpu_ids = list()

        for gpu_id in available_gpu_ids:
            if gpu_id not in realtime_available_gpu_ids:
                unavailable_gpu_ids.append(gpu_id)

        trimmed_gpu_ids = list(set(available_gpu_ids) - set(unavailable_gpu_ids))
        return trimmed_gpu_ids.copy()

    @staticmethod
    def release_partial_job_gpu(run_id, device_id, release_gpu_ids):
        """
        In the deployment phase, if scale in or update, we need to release the gpu ids for the partial job.
        """
        ComputeCacheManager.get_instance().set_redis_params()
        # Reversely find the master (launch) device id and release the gpu ids
        with ComputeCacheManager.get_instance().lock(
                ComputeCacheManager.get_instance().get_gpu_cache().get_run_lock_key(run_id)
        ):
            edge_device_id, model_master_device_id, model_slave_device_id = \
                ComputeCacheManager.get_instance().get_gpu_cache().get_edge_model_id_map(run_id)
            if edge_device_id is None:
                edge_device_id = device_id

        with ComputeCacheManager.get_instance().lock(
                ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_lock_key(
                    edge_device_id, JobRunnerUtils.STATIC_RUN_LOCK_KEY_SUFFIX)
        ):
            run_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_gpu_ids(device_id,
                                                                                                    run_id)

            if not run_gpu_ids:
                # Arrive here means this run is a rollback run, the reason that the run_gpu_ids is None is that
                # the run_id is the original run_id, not the inner_id.
                logging.info(f"Run {run_id} is None. Either it is already released or not occupied.")
                return

            remain_gpu_ids = [gpu_id for gpu_id in run_gpu_ids if gpu_id not in release_gpu_ids]

            # Update the available gpu ids
            with ComputeCacheManager.get_instance().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_device_lock_key(edge_device_id)
            ):
                # Set global available gpu ids
                available_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_available_gpu_ids(
                    edge_device_id)
                available_gpu_ids.extend(release_gpu_ids.copy())
                available_gpu_ids = list(dict.fromkeys(available_gpu_ids))
                ComputeCacheManager.get_instance().get_gpu_cache().set_device_available_gpu_ids(
                    edge_device_id, available_gpu_ids)

                # Set this run gpu ids
                ComputeCacheManager.get_instance().get_gpu_cache().set_device_run_gpu_ids(
                    device_id, run_id, remain_gpu_ids)

                # Set this run gpu num
                ComputeCacheManager.get_instance().get_gpu_cache().set_device_run_num_gpus(
                    device_id, run_id, len(remain_gpu_ids))

                logging.info(f"Run {run_id} released partial gpu ids: {release_gpu_ids}")

    def release_gpu_ids(self, run_id, device_id):
        edge_device_id = None
        original_run_id = None
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            with ComputeCacheManager.get_instance().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_run_lock_key(run_id)
            ):
                original_run_id = ComputeCacheManager.get_instance().get_gpu_cache().get_endpoint_run_id_map(run_id)
                edge_device_id, model_master_device_id, model_slave_device_id = \
                    ComputeCacheManager.get_instance().get_gpu_cache().get_edge_model_id_map(run_id)
                if edge_device_id is None or edge_device_id == 'None':
                    edge_device_id = device_id

            with ComputeCacheManager.get_instance().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_lock_key(
                        edge_device_id, JobRunnerUtils.STATIC_RUN_LOCK_KEY_SUFFIX)
            ):
                run_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_gpu_ids(device_id,
                                                                                                        run_id)
                if not run_gpu_ids:
                    logging.info(f"Run {run_id} is None. Either it is already released or not occupied.")
                    return

                with ComputeCacheManager.get_instance().lock(
                        ComputeCacheManager.get_instance().get_gpu_cache().get_device_lock_key(edge_device_id)
                ):
                    available_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_available_gpu_ids(
                        edge_device_id)
                    available_gpu_ids.extend(run_gpu_ids.copy())
                    available_gpu_ids = list(dict.fromkeys(available_gpu_ids))
                    ComputeCacheManager.get_instance().get_gpu_cache().set_device_available_gpu_ids(
                        edge_device_id, available_gpu_ids)

                    ComputeCacheManager.get_instance().get_gpu_cache().set_device_run_gpu_ids(device_id, run_id, [])

        except Exception as e:
            logging.error(f"Exception {e} occurred while releasing gpu ids. Traceback: {traceback.format_exc()}")
            pass

        if edge_device_id is not None:
            from fedml.core import mlops
            released_run_id = run_id if original_run_id is None else original_run_id
            logging.info(f"[run/device][{released_run_id}/{edge_device_id}] notify MLOps to release gpu resources.")
            mlops.release_resources(released_run_id, edge_device_id)

    def get_device_run_gpu_ids(self, device_id, run_id):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            with ComputeCacheManager.get_instance().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_lock_key(device_id, run_id)
            ):
                gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_run_gpu_ids(device_id, run_id)
                return gpu_ids
        except Exception as e:
            logging.error(f"Exception {e} occurred while getting device run gpu ids. "
                          f"Traceback: {traceback.format_exc()}")
            return []

    @staticmethod
    def get_available_gpu_id_list(device_id):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            with ComputeCacheManager.get_instance().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_device_lock_key(device_id)
            ):

                # Get the available GPU list from the cache
                available_gpu_ids = ComputeCacheManager.get_instance().get_gpu_cache().get_device_available_gpu_ids(
                    device_id)

                # If the available GPU list is not in the cache, set it to the current system available GPU list
                if available_gpu_ids is None:
                    # Get realtime GPU availability list from the system
                    gpu_ids = JobRunnerUtils.get_realtime_gpu_available_ids().copy()
                    ComputeCacheManager.get_instance().get_gpu_cache().set_device_available_gpu_ids(device_id, gpu_ids)
                    available_gpu_ids = gpu_ids
            return available_gpu_ids

        except Exception as e:
            logging.error(
                f"Exception {e} occurred while getting available GPU list. Traceback: {traceback.format_exc()}")
            return []

    @staticmethod
    def reset_available_gpu_id_list(device_id):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            with ComputeCacheManager.get_instance().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_device_lock_key(device_id)
            ):
                current_available_gpu_ids = JobRunnerUtils.get_realtime_gpu_available_ids().copy()
                ComputeCacheManager.get_instance().get_gpu_cache().set_device_available_gpu_ids(device_id,
                                                                                                current_available_gpu_ids)
                gpu_list = sys_utils.get_gpu_list()
                ComputeCacheManager.get_instance().get_gpu_cache().set_device_total_num_gpus(device_id, len(gpu_list))
        except Exception as e:
            logging.error(f"Exception {e} occurred while resetting available GPU list. "
                          f"Traceback: {traceback.format_exc()}")
            pass

    @staticmethod
    def get_realtime_gpu_available_ids():
        gpu_list = sys_utils.get_gpu_list()
        gpu_count = len(gpu_list)
        realtime_available_gpu_ids = sys_utils.get_available_gpu_id_list(limit=gpu_count)
        return realtime_available_gpu_ids

    @staticmethod
    def get_gpu_list_and_realtime_gpu_available_ids() -> (List[dict], List[int]):
        gpu_list = sys_utils.get_gpu_list()
        gpu_count = len(gpu_list)
        realtime_available_gpu_ids = sys_utils.get_available_gpu_id_list(limit=gpu_count)
        return gpu_list, realtime_available_gpu_ids

    @staticmethod
    def create_instance_from_dict(data_class, input_dict: {}):

        # Get the fields of the data class
        data_class_fields = fields(data_class)

        # Create an instance of the data class
        instance = data_class()

        # Set attributes based on input_dict with type checking
        for field in data_class_fields:
            if field.name in input_dict:
                input_value = input_dict[field.name]

                # Perform type checking
                if not isinstance(input_value, field.type):
                    raise TypeError(
                        f"Type mismatch for field '{field.name}'. Expected {field.type}, got {type(input_value)}.")

                setattr(instance, field.name, input_value)

        return instance

    @staticmethod
    def generate_bootstrap_commands(bootstrap_script_path, bootstrap_script_dir, bootstrap_script_file):
        if os.path.exists(bootstrap_script_path):
            bootstrap_stat = os.stat(bootstrap_script_path)
            if platform.system() == 'Windows':
                os.chmod(bootstrap_script_path,
                         bootstrap_stat.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                bootstrap_scripts = "{}".format(bootstrap_script_path)
            else:
                os.chmod(bootstrap_script_path,
                         bootstrap_stat.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                bootstrap_scripts = "cd {}; ./{}".format(
                    bootstrap_script_dir, os.path.basename(bootstrap_script_file))

            bootstrap_scripts = str(bootstrap_scripts).replace('\\', os.sep).replace('/', os.sep)
            shell_cmd_list = list()
            shell_cmd_list.append(bootstrap_scripts)
            return shell_cmd_list

    @staticmethod
    def generate_job_execute_commands(run_id, edge_id, version,
                                      package_type, executable_interpreter, entry_file_full_path,
                                      conf_file_object, entry_args, assigned_gpu_ids,
                                      job_api_key, client_rank, scheduler_match_info=None,
                                      cuda_visible_gpu_ids_str=None):
        shell_cmd_list = list()
        entry_commands_origin = list()

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
        entry_commands.insert(0,
                              f"{export_cmd} FEDML_ENV_LOCAL_ON_PREMISE_PLATFORM_HOST={fedml.get_local_on_premise_platform_host()}\n")
        entry_commands.insert(0,
                              f"{export_cmd} FEDML_ENV_LOCAL_ON_PREMISE_PLATFORM_PORT={fedml.get_local_on_premise_platform_port()}\n")
        if job_api_key is not None and str(job_api_key).strip() != "":
            random_out = sys_utils.random2(job_api_key, "FEDML@88119999GREAT")
            random_list = random_out.split("FEDML_NEXUS@")
            entry_commands.insert(0, f"{export_cmd} FEDML_RUN_API_KEY={random_list[1]}\n")

        # TODO: Remove adding this command entirely once we fully retire running launch on bare metal
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
            else:
                raise Exception(f"Unsupported entry file type: {entry_file_full_path}")
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
    def generate_launch_docker_command(docker_args: DockerArgs, run_id: int, edge_id: int,
                                       unzip_package_path: str, executable_interpreter: str, entry_file_full_path: str,
                                       bootstrap_cmd_list, cuda_visible_gpu_ids_str=None,
                                       image_pull_policy: str = None) -> List[str]:

        shell_command = list()

        docker_client = JobRunnerUtils.get_docker_client(docker_args=docker_args)

        ContainerUtils.get_instance().pull_image_with_policy(image_pull_policy, docker_args.image, client=docker_client)

        container_name = JobRunnerUtils.get_run_container_name(run_id)
        JobRunnerUtils.remove_run_container_if_exists(container_name, docker_client)

        docker_command = ["docker", "run", "-t", "--rm", "--name", f"{container_name}"]

        # Remove "export CUDA_VISIBLE_DEVICES=" from entry file and add as docker command instead:
        if cuda_visible_gpu_ids_str is not None:
            JobRunnerUtils.remove_cuda_visible_devices_lines(entry_file_full_path)
            # docker command expects device ids in such format: '"device=0,2,3"'
            device_str = f'"device={cuda_visible_gpu_ids_str}"'
            if not run_docker_without_gpu:
                docker_command.extend(["--gpus", f"'{device_str}'"])

        # Add Port Mapping
        for port in docker_args.ports:
            docker_command.extend(["-p", f"0:{port}"])

        # Mount Volumes
        home_dir = os.path.expanduser("~")
        log_file = "{}/.fedml/{}/fedml/logs/fedml-run-{}-edge-{}.log".format(
            home_dir, ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME, str(run_id), str(edge_id)
        )

        volumes = [log_file, unzip_package_path]
        for volume in volumes:
            docker_command.extend(["-v", f"{volume}:{volume}:rw"])

        # Add working directory
        working_directory = os.path.join(unzip_package_path, "fedml")
        docker_command.extend(["-w", working_directory])

        # Add image
        docker_command.extend(["--entrypoint", executable_interpreter])

        # Add image
        docker_command.append(docker_args.image)

        # Add entry command
        docker_command.append("-c")
        command_list = []
        if bootstrap_cmd_list:
            command_list.extend(bootstrap_cmd_list[0].split("; "))
        command_list.extend([f"chmod +x {entry_file_full_path}", f"{entry_file_full_path}"])
        cmd = " && ".join(command_list)
        docker_command.append(f'"{cmd}"')

        # Generate docker command to be executed in shell
        shell_command.append(" ".join(docker_command))

        return shell_command

    @staticmethod
    def get_run_container_name(run_id: int) -> str:
        container_prefix = f"{SchedulerConstants.FEDML_DEFAULT_LAUNCH_CONTAINER_PREFIX}"
        container_name = f"{container_prefix}__{run_id}"
        return container_name

    @staticmethod
    def get_docker_client(docker_args: DockerArgs) -> DockerClient:
        try:
            client = docker.from_env()
            if docker_args.username != "" and docker_args.registry != "":
                client.login(username=docker_args.username, password=docker_args.password, registry=docker_args.registry)
        except Exception as e:
            raise Exception(f"Failed to connect to the docker daemon, please ensure that you have "
                            f"installed Docker Desktop or Docker Engine, and the docker is running. Exception {e}")
        return client

    @staticmethod
    def remove_run_container_if_exists(container_name: str, client: DockerClient):

        try:
            exist_container_obj = client.containers.get(container_name)
            logging.info(f"Container {container_name} found")
        except docker.errors.NotFound:
            logging.info(f"Container {container_name} not found")
            exist_container_obj = None
        except docker.errors.APIError:
            raise Exception("Failed to get the container object")

        if exist_container_obj is not None:
            client.api.remove_container(exist_container_obj.id, v=True, force=True)
            logging.info(f"Container {container_name} removed")

    @staticmethod
    def remove_cuda_visible_devices_lines(file_path):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Remove lines containing 'export CUDA_VISIBLE_DEVICES='
            modified_lines = [line for line in lines if 'export CUDA_VISIBLE_DEVICES=' not in line]

            with open(file_path, 'w') as f:
                f.writelines(modified_lines)

            logging.info(f"Lines containing 'export CUDA_VISIBLE_DEVICES=' removed successfully from {file_path}")
        except FileNotFoundError:
            logging.error(f"Error: File '{file_path}' not found.")
        except Exception as e:
            logging.error(f"An error occurred while removing cuda visible devices from {file_path} : {e}")

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
            config_key = f"{config_key_path}_{config_key}".upper() if config_key_path else str(config_key).upper()
            if isinstance(config_value, dict):
                JobRunnerUtils.get_env_from_dict(
                    export_cmd, config_value, export_env_command_list=export_env_command_list,
                    env_name_value_map=env_name_value_map, config_key_path=config_key
                )
            else:
                env_name = f"FEDML_ENV_{config_key}"
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

    @staticmethod
    def parse_job_type(running_json):
        if running_json is None:
            return None
        running_json_obj = json.loads(running_json) if not isinstance(running_json, dict) else running_json
        run_config = running_json_obj.get("run_config", {})
        parameters = run_config.get("parameters", {})
        job_yaml = parameters.get("job_yaml", {})
        job_type = job_yaml.get("job_type", None)
        job_type = job_yaml.get("task_type",
                                SchedulerConstants.JOB_TASK_TYPE_TRAIN) if job_type is None else job_type
        model_config = running_json_obj.get("model_config", None)
        if model_config is not None:
            job_type = SchedulerConstants.JOB_TASK_TYPE_DEPLOY
        return job_type

    @staticmethod
    def get_job_type_from_run_id(run_id: str) -> str:
        job_type = None
        try:
            job_obj = FedMLClientDataInterface.get_instance().get_job_by_id(run_id)
            if job_obj is not None:
                job_json = json.loads(job_obj.running_json)
                run_config = job_json.get("run_config", {})
                run_params = run_config.get("parameters", {})
                job_yaml = run_params.get("job_yaml", {})
                job_type = job_yaml.get("job_type", None)
                job_type = job_yaml.get("task_type",
                                        SchedulerConstants.JOB_TASK_TYPE_TRAIN) if job_type is None else job_type
        except Exception as e:
            logging.debug(f"Failed to get job obj with Exception {e}. Traceback: {traceback.format_exc()}")
        return job_type
