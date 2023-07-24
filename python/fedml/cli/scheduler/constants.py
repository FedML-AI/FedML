import os
from os.path import expanduser

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

    @staticmethod
    def get_fedml_home_dir(is_client=True):
        home_dir = expanduser("~")
        fedml_home_dir = os.path.join(home_dir,
                                      Constants.FEDML_DEVICE_RUNNER_TYPE_CLIENT
                                      if is_client else Constants.FEDML_DEVICE_RUNNER_TYPE_SERVER)
        if not os.path.exists(fedml_home_dir):
            os.makedirs(fedml_home_dir, exist_ok=True)
        return fedml_home_dir
