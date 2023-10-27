import os
import platform

from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.comm_utils.sys_utils import get_python_program
from fedml.computing.scheduler.model_scheduler.device_model_cache import FedMLModelCache
from fedml.core.common.singleton import Singleton
import threading


class JobRunnerUtils(Singleton):
    FEDML_SUPPORTED_ENVIRONMENT_VARIABLES = ["$FEDML_MODEL_NAME", "$FEDML_MODEL_CACHE_PATH",
                                             "$FEDML_MODEL_INPUT_DIM", "$FEDML_MODEL_OUTPUT_DIM",
                                             "$FEDML_DATASET_NAME", "$FEDML_DATASET_PATH", "$FEDML_DATASET_TYPE",
                                             "$FEDML_NODE_0_ADDR", "$FEDML_NODE_0_PORT", "$FEDML_NUM_NODES",
                                             "$CUDA_VISIBLE_DEVICES"]

    def __init__(self):
        if not hasattr(self, "run_id_to_gpu_ids_map"):
            self.run_id_to_gpu_ids_map = dict()
        if not hasattr(self, "available_gpu_ids"):
            self.available_gpu_ids = list()
            self.available_gpu_ids = self.get_realtime_gpu_available_ids().copy()
        if not hasattr(self, "lock_available_gpu_ids"):
            self.lock_available_gpu_ids = threading.Lock()

    @staticmethod
    def get_instance():
        return JobRunnerUtils()

    def occupy_gpu_ids(self, run_id, request_gpu_num, inner_id=None):
        self.lock_available_gpu_ids.acquire()

        available_gpu_count = len(self.available_gpu_ids)
        request_gpu_num = 0 if request_gpu_num is None else request_gpu_num
        matched_gpu_num = min(available_gpu_count, request_gpu_num)
        if matched_gpu_num <= 0:
            self.lock_available_gpu_ids.release()
            return None

        matched_gpu_ids = map(lambda x: str(x), self.available_gpu_ids[0:matched_gpu_num])
        cuda_visiable_gpu_ids_str = ",".join(matched_gpu_ids)

        self.run_id_to_gpu_ids_map[str(run_id)] = self.available_gpu_ids[0:matched_gpu_num].copy()
        self.available_gpu_ids = self.available_gpu_ids[matched_gpu_num:].copy()
        self.available_gpu_ids = list(dict.fromkeys(self.available_gpu_ids))

        if inner_id is not None:
            FedMLModelCache.get_instance().set_redis_params()
            FedMLModelCache.get_instance().set_end_point_gpu_resources(
                inner_id, matched_gpu_num, cuda_visiable_gpu_ids_str)

        self.lock_available_gpu_ids.release()

        return cuda_visiable_gpu_ids_str

    def release_gpu_ids(self, run_id):
        self.lock_available_gpu_ids.acquire()
        occupy_gpu_id_list = self.run_id_to_gpu_ids_map.get(str(run_id), [])
        self.available_gpu_ids.extend(occupy_gpu_id_list.copy())
        self.available_gpu_ids = list(dict.fromkeys(self.available_gpu_ids))
        self.lock_available_gpu_ids.release()

    def get_available_gpu_id_list(self):
        self.lock_available_gpu_ids.acquire()
        ret_gpu_ids = self.available_gpu_ids.copy()
        self.lock_available_gpu_ids.release()
        return ret_gpu_ids

    def get_realtime_gpu_available_ids(self):
        gpu_list = sys_utils.get_gpu_list()
        gpu_count = len(gpu_list)
        realtime_available_gpu_ids = sys_utils.get_available_gpu_id_list(limit=gpu_count)
        return realtime_available_gpu_ids

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
        export_env_cmd_list, env_name_value_map = JobRunnerUtils.parse_config_args_as_env_variables(
            export_cmd, conf_file_object, job_yaml=job_yaml)

        # Generate the export env list about scheduler matching info for publishing environment variables
        export_env_cmd_list_for_match, env_name_value_map_for_match = \
            JobRunnerUtils.assign_matched_resources_to_run_and_generate_envs(
                run_id, export_cmd, scheduler_match_info
            )

        # Replace entry commands with environment variable values
        entry_commands = JobRunnerUtils.replace_entry_command_with_env_variable(
            entry_commands_origin, env_name_value_map
        )
        entry_commands = JobRunnerUtils.replace_entry_command_with_env_variable(
            entry_commands, env_name_value_map_for_match
        )

        # Replace entry arguments with environment variable values
        entry_args = JobRunnerUtils.replace_entry_args_with_env_variable(entry_args, env_name_value_map)

        # Add the export env list to the entry commands
        if len(export_env_cmd_list) > 0:
            entry_commands.extend(export_env_cmd_list)
        for match_cmd in export_env_cmd_list_for_match:
            entry_commands.insert(0, match_cmd)

        # Add general environment variables
        entry_commands.insert(0, f"{export_cmd} FEDML_CURRENT_EDGE_ID={edge_id}\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_CURRENT_RUN_ID={run_id}\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_CURRENT_VERSION={version}\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_ENV_VERSION={version}\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_USING_MLOPS=true\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_CLIENT_RANK={client_rank}\n")
        if job_api_key is not None and str(job_api_key).strip() != "":
            entry_commands.insert(0, f"{export_cmd} FEDML_RUN_API_KEY={job_api_key}\n")
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
            for env_name in JobRunnerUtils.FEDML_SUPPORTED_ENVIRONMENT_VARIABLES:
                env_value = env_name_value_map.get(env_name, None)
                if env_value is None:
                    continue
                entry_cmd = entry_cmd.replace(env_name, str(env_value))

            entry_commands_replaced.append(entry_cmd)

        return entry_commands_replaced

    @staticmethod
    def replace_entry_args_with_env_variable(entry_args, env_name_value_map):
        if entry_args is None:
            return ""
        for env_name in JobRunnerUtils.FEDML_SUPPORTED_ENVIRONMENT_VARIABLES:
            env_value = env_name_value_map.get(env_name, None)
            if env_value is None:
                continue
            entry_args = entry_args.replace(env_name, str(env_value))

        return entry_args

    @staticmethod
    def parse_config_args_as_env_variables(export_cmd, run_params, job_yaml=None):
        model_args = run_params.get("fedml_model_args", None)
        if model_args is None:
            model_args = job_yaml.get("fedml_model_args", {}) if job_yaml is not None else dict()

        data_args = run_params.get("fedml_data_args", None)
        if data_args is None:
            data_args = job_yaml.get("fedml_data_args", {}) if job_yaml is not None else dict()

        model_name = model_args.get("model_name", None)
        model_cache_path = model_args.get("model_cache_path", None)
        input_dim = model_args.get("input_dim", None)
        output_dim = model_args.get("output_dim", None)
        dataset_name = data_args.get("dataset_name", None)
        dataset_path = data_args.get("dataset_path", None)
        dataset_type = data_args.get("dataset_type", None)

        export_env_command_list = list()
        env_name_value_map = dict()

        if model_name is not None and str(model_name).strip() != "":
            export_env_command_list.append(f"{export_cmd} FEDML_MODEL_NAME={model_name}\n")
            env_name_value_map["$FEDML_MODEL_NAME"] = model_name

        if model_cache_path is not None and str(model_cache_path).strip() != "":
            export_env_command_list.append(f"{export_cmd} FEDML_MODEL_CACHE_PATH={model_cache_path}\n")
            env_name_value_map["$FEDML_MODEL_CACHE_PATH"] = model_cache_path

        if input_dim is not None and str(input_dim).strip() != "":
            export_env_command_list.append(f"{export_cmd} FEDML_MODEL_INPUT_DIM={input_dim}\n")
            env_name_value_map["$FEDML_MODEL_INPUT_DIM"] = input_dim

        if output_dim is not None and str(output_dim).strip() != "":
            export_env_command_list.append(f"{export_cmd} MODEL_OUTPUT_DIM={output_dim}\n")
            env_name_value_map["$MODEL_OUTPUT_DIM"] = output_dim

        if dataset_name is not None and str(dataset_name).strip() != "":
            export_env_command_list.append(f"{export_cmd} FEDML_DATASET_NAME={dataset_name}\n")
            env_name_value_map["$FEDML_DATASET_NAME"] = dataset_name

        if dataset_path is not None and str(dataset_path).strip() != "":
            export_env_command_list.append(f"{export_cmd} FEDML_DATASET_PATH={dataset_path}\n")
            env_name_value_map["$FEDML_DATASET_PATH"] = dataset_path

        if dataset_type is not None and str(dataset_type).strip() != "":
            export_env_command_list.append(f"{export_cmd} FEDML_DATASET_TYPE={dataset_type}\n")
            env_name_value_map["$FEDML_DATASET_TYPE"] = dataset_type

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
            env_name_value_map["$FEDML_NODE_0_ADDR"] = master_node_addr

        if master_node_port is not None and str(master_node_port).strip() != "":
            export_env_command_list.append(f"{export_cmd} FEDML_NODE_0_PORT={master_node_port}\n")
            env_name_value_map["$FEDML_NODE_0_PORT"] = master_node_port

        if num_nodes is not None and str(num_nodes).strip() != "":
            export_env_command_list.append(f"{export_cmd} FEDML_NUM_NODES={num_nodes}\n")
            env_name_value_map["$FEDML_NUM_NODES"] = num_nodes

        return export_env_command_list, env_name_value_map
