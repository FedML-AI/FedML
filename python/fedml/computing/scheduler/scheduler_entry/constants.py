import datetime
import os
from os.path import expanduser

import yaml
from fedml.core.common.singleton import Singleton


class Constants(Singleton):
    FEDML_MLOPS_BUILD_PRE_IGNORE_LIST = 'dist-packages,client-package.zip,server-package.zip,__pycache__,*.pyc,*.git, *venv'
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

    ERROR_CODE_MODEL_CREATE_FAILED = 1
    ERROR_CODE_MODEL_ADD_FILES_FAILED = 2
    ERROR_CODE_MODEL_BUILD_FAILED = 3

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

    LAUNCH_JOB_DEFAULT_FOLDER_NAME = "fedml_job_pack"
    LAUNCH_JOB_DEFAULT_ENTRY_NAME = "fedml_job_entry_pack.sh"
    LAUNCH_SERVER_JOB_DEFAULT_ENTRY_NAME = "fedml_server_job_entry_pack.sh"
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

    JOB_START_STATUS_MATCHED = "MATCHED"
    JOB_START_STATUS_JOB_URL_ERROR = "ERROR_JOB_URL"
    JOB_START_STATUS_INVALID = "INVALID"
    JOB_START_STATUS_BLOCKED = "BLOCKED"
    JOB_START_STATUS_QUEUED = "QUEUED"
    JOB_START_STATUS_BIND_CREDIT_CARD_FIRST = "BIND_CREDIT_CARD_FIRST"
    JOB_START_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED = "QUERY_CREDIT_CARD_BINDING_STATUS_FAILED"

    TIME_PER_HOUR_TO_MS = 3600000.0

    JOB_FRAMEWORK_TYPE_FEDML = "fedml"
    JOB_FRAMEWORK_TYPE_DEEPSPEED = "deepspeed"
    JOB_FRAMEWORK_TYPE_PYTORCH = "pytorch"
    JOB_FRAMEWORK_TYPE_TENSORFLOW = "tensorflow"
    JOB_FRAMEWORK_TYPE_MXNET = "mxnet"
    JOB_FRAMEWORK_TYPE_GENERAL = "general"

    JOB_TASK_TYPE_TRAIN = "train"
    JOB_TASK_TYPE_SERVE = "serve"
    JOB_TASK_TYPE_DEV_ENV = "dev-environment"

    JOB_DEVICE_TYPE_CPU = "CPU"
    JOB_DEVICE_TYPE_GPU = "GPU"
    JOB_DEVICE_TYPE_HYBRID = "hybrid"

    CHECK_MARK_STRING = chr(8730)

    JOB_LOG_PAGE_SIZE = 100

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

    @staticmethod
    def format_time_trimmed_tz(date_time_tz):
        try:
            formatted_time = datetime.datetime.strptime(date_time_tz, "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
            return formatted_time
        except Exception as e:
            pass

        try:
            formatted_time = datetime.datetime.strptime(date_time_tz, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d "
                                                                                                     "%H:%M:%S")
            return formatted_time
        except Exception as e:
            return date_time_tz

    @staticmethod
    def get_current_time_zone():
        tz_info = "Etc/UTC"
        try:
            import tzlocal
            tz_info = tzlocal.get_localzone()
        except Exception as e:
            pass
        return str(tz_info)
