import os
import platform

from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.comm_utils.sys_utils import get_python_program


class JobRunnerUtils:
    FEDML_SUPPORTED_ENVIRONMENT_VARIABLES = ["$FEDML_MODEL_NAME", "$FEDML_MODEL_CACHE_PATH",
                                             "$FEDML_MODEL_INPUT_DIM", "$FEDML_MODEL_OUTPUT_DIM",
                                             "$FEDML_DATASET_NAME", "$FEDML_DATASET_PATH", "$FEDML_DATASET_TYPE"]

    @staticmethod
    def generate_job_execute_commands(run_id, edge_id, version,
                                      package_type, executable_interpreter, entry_file_full_path,
                                      conf_file_object, entry_args, assigned_gpu_ids,
                                      job_api_key, client_rank, job_yaml=None, request_gpu_num=None):
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
        export_env_list, env_value_map = JobRunnerUtils.parse_config_args_as_env_variables(
            export_cmd, conf_file_object, job_yaml=job_yaml)

        # Replace entry commands with environment variable values
        entry_commands = JobRunnerUtils.replace_entry_command_with_env_variable(entry_commands_origin, env_value_map)

        # Replace entry arguments with environment variable values
        entry_args = JobRunnerUtils.replace_entry_args_with_env_variable(entry_args, env_value_map)

        # Add the export env list to the entry commands
        if len(export_env_list) > 0:
            entry_commands.extend(export_env_list)

        # Add general environment variables
        entry_commands.insert(0, f"{export_cmd} FEDML_CURRENT_EDGE_ID={edge_id}\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_CURRENT_RUN_ID={run_id}\n")
        if assigned_gpu_ids is None or str(assigned_gpu_ids).strip() == "":
            assigned_gpu_ids = JobRunnerUtils.apply_gpu_ids(request_gpu_num)
        if assigned_gpu_ids is not None and str(assigned_gpu_ids).strip() != "":
            entry_commands.insert(0, f"{export_cmd} CUDA_VISIBLE_DEVICES={assigned_gpu_ids}\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_CURRENT_VERSION={version}\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_USING_MLOPS=true\n")
        entry_commands.insert(0, f"{export_cmd} FEDML_CLIENT_RANK={client_rank}\n")
        if job_api_key is not None and str(job_api_key).strip() != "":
            entry_commands.insert(0, f"{export_cmd} FEDML_RUN_API_KEY={job_api_key}\n")

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
    def replace_entry_command_with_env_variable(entry_commands, env_value_map):
        entry_commands_replaced = list()
        for entry_cmd in entry_commands:
            for env_name in JobRunnerUtils.FEDML_SUPPORTED_ENVIRONMENT_VARIABLES:
                env_value = env_value_map.get(env_name, None)
                if env_value is None:
                    continue
                entry_cmd = entry_cmd.replace(env_name, env_value)

            entry_commands_replaced.append(entry_cmd)

        return entry_commands_replaced

    @staticmethod
    def replace_entry_args_with_env_variable(entry_args, env_value_map):
        if entry_args is None:
            return ""
        for env_name in JobRunnerUtils.FEDML_SUPPORTED_ENVIRONMENT_VARIABLES:
            env_value = env_value_map.get(env_name, None)
            if env_value is None:
                continue
            entry_args = entry_args.replace(env_name, env_value)

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

        export_env_list = list()
        env_value_map = dict()

        if model_name is not None and str(model_name).strip() != "":
            export_env_list.append(f"{export_cmd} FEDML_MODEL_NAME={model_name}\n")
            env_value_map["$FEDML_MODEL_NAME"] = model_name

        if model_cache_path is not None and str(model_cache_path).strip() != "":
            export_env_list.append(f"{export_cmd} FEDML_MODEL_CACHE_PATH={model_cache_path}\n")
            env_value_map["$FEDML_MODEL_CACHE_PATH"] = model_cache_path

        if input_dim is not None and str(input_dim).strip() != "":
            export_env_list.append(f"{export_cmd} FEDML_MODEL_INPUT_DIM={input_dim}\n")
            env_value_map["$FEDML_MODEL_INPUT_DIM"] = input_dim

        if output_dim is not None and str(output_dim).strip() != "":
            export_env_list.append(f"{export_cmd} MODEL_OUTPUT_DIM={output_dim}\n")
            env_value_map["$MODEL_OUTPUT_DIM"] = output_dim

        if dataset_name is not None and str(dataset_name).strip() != "":
            export_env_list.append(f"{export_cmd} FEDML_DATASET_NAME={dataset_name}\n")
            env_value_map["$FEDML_DATASET_NAME"] = dataset_name

        if dataset_path is not None and str(dataset_path).strip() != "":
            export_env_list.append(f"{export_cmd} FEDML_DATASET_PATH={dataset_path}\n")
            env_value_map["$FEDML_DATASET_PATH"] = dataset_path

        if dataset_type is not None and str(dataset_type).strip() != "":
            export_env_list.append(f"{export_cmd} FEDML_DATASET_TYPE={dataset_type}\n")
            env_value_map["$FEDML_DATASET_TYPE"] = dataset_type

        return export_env_list, env_value_map

    @staticmethod
    def apply_gpu_ids(request_gpu_num):
        gpu_list = sys_utils.get_gpu_list()
        gpu_count = len(gpu_list)
        available_gpu_ids = sys_utils.get_available_gpu_id_list(limit=gpu_count)
        available_gpu_count = len(available_gpu_ids)
        request_gpu_num = 0 if request_gpu_num is None else request_gpu_num
        matched_gpu_num = min(available_gpu_count, request_gpu_num)
        if matched_gpu_num <= 0:
            return None
        matched_gpu_ids = map(lambda x: str(x), available_gpu_ids[0:matched_gpu_num])
        return ",".join(matched_gpu_ids)
