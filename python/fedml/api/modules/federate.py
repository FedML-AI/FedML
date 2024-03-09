import os
import shutil

from fedml.api.modules.constants import ModuleConstants
from fedml.computing.scheduler.comm_utils.sys_utils import generate_yaml_doc
from fedml.computing.scheduler.comm_utils.yaml_utils import load_yaml_config
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.scheduler_entry.constants import Constants as SchedulerEntryConstants
import fedml.api.modules.build
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager


def build(is_built_client_package, source_folder, entry_point, entry_args, config_folder, dest_folder, ignore,
          model_name, model_cache_path, input_dim, output_dim, dataset_name, dataset_type, dataset_path):
    # Check the config file
    config_file_path = os.path.join(config_folder, ModuleConstants.FEDML_CONFIG_YAML_FILE)
    if not os.path.exists(config_file_path):
        print(f"Please make sure the following config file exists. \n{config_file_path}")
        return

    # Load the config yaml file
    config_dict = load_yaml_config(config_file_path)
    if config_dict is None:
        config_dict = dict()

    # Generate the entry arguments
    if entry_args is not None and str(entry_args).strip() != "":
        config_dict["fedml_entry_args"] = dict()
        config_dict["fedml_entry_args"]["arg_items"] = f"{entry_args}"

    # Save the updated config object into the config yaml file
    generate_yaml_doc(config_dict, config_file_path)

    # Build the package based on the updated config file
    fedml.api.modules.build.build(
        ModuleConstants.PLATFORM_NAME_LAUNCH,
        ModuleConstants.TRAIN_BUILD_PACKAGE_CLIENT_TYPE if is_built_client_package
        else ModuleConstants.TRAIN_BUILD_PACKAGE_SERVER_TYPE,
        source_folder, entry_point, config_folder, dest_folder, ignore,
        package_type=SchedulerConstants.JOB_PACKAGE_TYPE_FEDERATE)


def build_with_job_yaml(job_yaml_file, dest_folder=None):
    job_config, app_config, client_package, server_package = FedMLLaunchManager.get_instance().prepare_launch(
        job_yaml_file)
    if client_package is not None and os.path.exists(client_package) is False:
        print("Build failed, please check your job yaml file.")
        return
    if server_package is not None and os.path.exists(server_package) is False:
        print("Build failed, please check your job yaml file for server.")
        return

    job_dir_path = os.path.dirname(job_yaml_file)
    if dest_folder is None or str(dest_folder).strip() == "":
        dest_folder = os.path.join(job_dir_path, ModuleConstants.FEDERATE_BUILD_DEFAULT_DEST_DIR_NAME)

    os.makedirs(dest_folder, exist_ok=True)
    if client_package is not None:
        dest_package = os.path.normpath(
            os.path.join(dest_folder, os.path.basename(client_package)))
        shutil.copyfile(client_package, dest_package)
        print(f"Your client package file is located at: {dest_package}")

    if server_package is not None:
        dest_package = os.path.normpath(
            os.path.join(dest_folder, os.path.basename(server_package)))
        shutil.copyfile(server_package, dest_package)
        print(f"Your server package file is located at: {dest_package}")

    bootstrap_sh_file = os.path.join(job_dir_path, SchedulerEntryConstants.BOOTSTRAP_FILE_NAME)
    bootstrap_bat_file = bootstrap_sh_file.rstrip(".sh") + ".bat"
    job_entry_bat_file = os.path.join(
        job_dir_path, SchedulerConstants.LAUNCH_JOB_DEFAULT_ENTRY_NAME.rstrip('.sh') + '.bat')
    job_entry_sh_file = os.path.join(job_dir_path, SchedulerConstants.LAUNCH_JOB_DEFAULT_ENTRY_NAME)
    server_job_entry_bat_file = os.path.join(
        job_dir_path, SchedulerConstants.LAUNCH_SERVER_JOB_DEFAULT_ENTRY_NAME.rstrip('.sh') + '.bat')
    server_job_entry_sh_file = os.path.join(
        job_dir_path, SchedulerConstants.LAUNCH_SERVER_JOB_DEFAULT_ENTRY_NAME)
    if os.path.exists(bootstrap_bat_file):
        os.remove(bootstrap_bat_file)
    if os.path.exists(bootstrap_sh_file):
        os.remove(bootstrap_sh_file)
    if os.path.exists(job_entry_bat_file):
        os.remove(job_entry_bat_file)
    if os.path.exists(job_entry_sh_file):
        os.remove(job_entry_sh_file)
    if os.path.exists(server_job_entry_bat_file):
        os.remove(server_job_entry_bat_file)
    if os.path.exists(server_job_entry_sh_file):
        os.remove(server_job_entry_sh_file)
