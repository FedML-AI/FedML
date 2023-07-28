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
    FEDML_PLATFORM_FALCON_STR = "falcon"
    FEDML_PLATFORM_OCTOPUS_TYPE = 1
    FEDML_PLATFORM_PARROT_TYPE = 2
    FEDML_PLATFORM_CHEETAH_TYPE = 3
    FEDML_PLATFORM_BEEHIVE_TYPE = 4
    FEDML_PLATFORM_FALCON_TYPE = 5

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
        elif platform_str == Constants.FEDML_PLATFORM_FALCON_STR:
            return Constants.FEDML_PLATFORM_FALCON_TYPE

        return Constants.FEDML_PLATFORM_FALCON_TYPE
