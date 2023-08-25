import os
from os.path import expanduser

import yaml
from fedml.core.common.singleton import Singleton


class Constants(Singleton):
    FEDML_MLOPS_BUILD_PRE_IGNORE_LIST = 'dist-packages,client-package.zip,server-package.zip,__pycache__,*.pyc,*.git'
    FEDML_PLATFORM_OCTOPUS_STR = "octopus"
    FEDML_PLATFORM_PARROT_STR = "parrot"
    FEDML_PLATFORM_CHEETAH_STR = "cheetah"
    FEDML_PLATFORM_BEEHIVE_STR = "beehive"
    FEDML_PLATFORM_LAUNCH_STR = "launch"
    FEDML_PLATFORM_OCTOPUS_TYPE = 1
    FEDML_PLATFORM_PARROT_TYPE = 2
    FEDML_PLATFORM_CHEETAH_TYPE = 3
    FEDML_PLATFORM_BEEHIVE_TYPE = 4
    FEDML_PLATFORM_LAUNCH_TYPE = 5

    FEDML_PACKAGE_BUILD_TARGET_TYPE_CLIENT = "client"
    FEDML_PACKAGE_BUILD_TARGET_TYPE_SERVER = "server"

    FEDML_DEVICE_RUNNER_TYPE_CLIENT = "fedml-client"
    FEDML_DEVICE_RUNNER_TYPE_SERVER = "fedml-server"

    FEDML_LAUNCH_JOB_TEMP_DIR = "tmp"

    BOOTSTRAP_FILE_NAME = "bootstrap.sh"
    STD_CONFIG_ENV_SECTION = "environment_args"
    STD_CONFIG_ENV_SECTION_BOOTSTRAP_KEY = "bootstrap"

    OS_PLATFORM_WINDOWS = 'Windows'

    LAUNCH_PARAMETER_JOB_YAML_KEY = "job_yaml"

    MLOPS_CLIENT_STATUS_NOT_STARTED = "NotStarted"
    MLOPS_CLIENT_STATUS_ACTIVE = "Active"
    MLOPS_CLIENT_STATUS_DONE = "Done"
    MLOPS_CLIENT_STATUS_STOPPED = "Stopped"

    LAUNCH_JOB_DEFAULT_FOLDER_NAME = "example_job"
    LAUNCH_JOB_DEFAULT_ENTRY_NAME = "example_entry.sh"
    LAUNCH_JOB_DEFAULT_CONF_FOLDER_NAME = "config"
    LAUNCH_JOB_DEFAULT_CONF_NAME = "fedml_config.yaml"

    GPU_BRAND_MAPPING_INDEX_NVIDIA = 0
    GPU_BRAND_MAPPING_INDEX_AMD = 1
    GPU_BRAND_MAPPING_INDEX_INTEL = 2
    GPU_BRAND_MAPPING_INDEX_OTHER = 3
    GPU_BRAND_MAPPING = {GPU_BRAND_MAPPING_INDEX_NVIDIA: "Nvidia",
                         GPU_BRAND_MAPPING_INDEX_AMD: "AMD",
                         GPU_BRAND_MAPPING_INDEX_INTEL: "Intel",
                         GPU_BRAND_MAPPING_INDEX_OTHER: "Other"}

    FEDML_DIR = "fedml"
    DATA_DIR = "data"
    SEC_KEY_DIR = "secret"
    SEC_KEY_FILE = "launch_secret"

    LAUNCH_APP_NAME_PREFIX = "FedMLLaunchApp"
    LAUNCH_PROJECT_NAME_DEFAULT = "default"

    JOB_START_STATUS_INVALID = "INVALID"
    JOB_START_STATUS_BLOCKED = "BLOCKED"
    JOB_START_STATUS_QUEUED = "QUEUED"

    TIME_PER_HOUR_TO_MS = 3600000.0

    @staticmethod
    def get_fedml_home_dir(is_client=True):
        home_dir = expanduser("~")
        fedml_home_dir = os.path.join(home_dir,
                                      Constants.FEDML_DEVICE_RUNNER_TYPE_CLIENT
                                      if is_client else Constants.FEDML_DEVICE_RUNNER_TYPE_SERVER)
        if not os.path.exists(fedml_home_dir):
            os.makedirs(fedml_home_dir, exist_ok=True)
        return fedml_home_dir

    @staticmethod
    def get_data_dir():
        data_dir = os.path.join(Constants.get_fedml_home_dir(), Constants.FEDML_DIR, Constants.DATA_DIR)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        return data_dir

    @staticmethod
    def get_secret_dir():
        secret_dir = os.path.join(Constants.get_data_dir(), Constants.SEC_KEY_DIR)
        if not os.path.exists(secret_dir):
            os.makedirs(secret_dir, exist_ok=True)
        return secret_dir

    @staticmethod
    def get_launch_secret_file():
        secret_file = os.path.join(Constants.get_secret_dir(), Constants.SEC_KEY_FILE)
        return secret_file

    @staticmethod
    def generate_yaml_doc(run_config_object, yaml_file):
        try:
            file = open(yaml_file, 'w', encoding='utf-8')
            yaml.dump(run_config_object, file)
            file.close()
        except Exception as e:
            pass

    @staticmethod
    def platform_str_to_type(platform_str):
        if platform_str == Constants.FEDML_PLATFORM_OCTOPUS_STR:
            return Constants.FEDML_PLATFORM_OCTOPUS_TYPE
        elif platform_str == Constants.FEDML_PLATFORM_PARROT_STR:
            return Constants.FEDML_PLATFORM_PARROT_TYPE
        elif platform_str == Constants.FEDML_PLATFORM_BEEHIVE_STR:
            return Constants.FEDML_PLATFORM_BEEHIVE_TYPE
        elif platform_str == Constants.FEDML_PLATFORM_CHEETAH_STR:
            return Constants.FEDML_PLATFORM_CHEETAH_TYPE
        elif platform_str == Constants.FEDML_PLATFORM_LAUNCH_STR:
            return Constants.FEDML_PLATFORM_LAUNCH_TYPE

        return Constants.FEDML_PLATFORM_LAUNCH_TYPE
