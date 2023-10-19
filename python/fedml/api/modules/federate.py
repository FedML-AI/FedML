import os

from fedml.api.modules.constants import ModuleConstants
from fedml.computing.scheduler.comm_utils.sys_utils import generate_yaml_doc
from fedml.computing.scheduler.comm_utils.yaml_utils import load_yaml_config
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
import fedml.api.modules.build


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

    # Generate the model args based on the input arguments
    if config_dict.get("fedml_model_args", None) is None:
        config_dict["fedml_model_args"] = dict()
    if model_name is not None and str(model_name).strip() != "":
        config_dict["fedml_model_args"]["model_name"] = model_name
    if model_cache_path is not None and str(model_cache_path).strip() != "":
        config_dict["fedml_model_args"]["model_cache_path"] = model_cache_path
    if input_dim is not None and str(input_dim).strip() != "":
        config_dict["fedml_model_args"]["input_dim"] = input_dim
    if output_dim is not None and str(output_dim).strip() != "":
        config_dict["fedml_model_args"]["output_dim"] = output_dim

    # Generate the data args based on the input arguments
    if config_dict.get("fedml_data_args", None) is None:
        config_dict["fedml_data_args"] = dict()
    if dataset_name is not None and str(dataset_name).strip() != "":
        config_dict["fedml_data_args"]["dataset_name"] = dataset_name
    if dataset_type is not None and str(dataset_type).strip() != "":
        config_dict["fedml_data_args"]["dataset_type"] = dataset_type
    if dataset_path is not None and str(dataset_path).strip() != "":
        config_dict["fedml_data_args"]["dataset_path"] = dataset_path

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
