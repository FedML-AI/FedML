import os
import platform
import shutil
import time
import uuid
from os.path import expanduser

import click
from fedml.computing.scheduler.comm_utils.sys_utils import daemon_ota_upgrade_with_version, check_fedml_is_latest_version
from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.comm_utils.yaml_utils import load_yaml_config
from fedml.computing.scheduler.comm_utils.platform_utils import platform_is_valid
from fedml.core.mlops import MLOpsUtils

from constants import Constants
from app_manager import FedMLAppManager
from job_manager import FedMLJobManager


class FedMLLaunchManager(Singleton):
    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLLaunchManager()

    def set_config_version(self, config_version):
        if config_version is not None:
            self.config_version = config_version

    def launch_job(self, yaml_file, user_api_key, mlops_platform_type,
                   device_server, device_edges,
                   no_confirmation=False):

        try:
            is_latest_version, _, _ = check_fedml_is_latest_version(configuration_env=self.config_version)
            if not is_latest_version:
                daemon_ota_upgrade_with_version(in_version=self.config_version)
                click.echo("Completed upgrading, please launch your job again.")
                exit(-1)
        except Exception as e:
            pass

        if not os.path.exists(yaml_file):
            click.echo(f"{yaml_file} can not be found. Please specify the full path of your job yaml file.")
            exit(-1)

        if os.path.dirname(yaml_file) == "":
            yaml_file = os.path.join(os.getcwd(), yaml_file)

        # Parse the job yaml file and regenerated application name if the job name is not given.
        self.parse_job_yaml(yaml_file)

        # Generate source, config and bootstrap related paths.
        platform_str = mlops_platform_type
        platform_type = Constants.platform_str_to_type(mlops_platform_type)
        client_server_type = Constants.FEDML_PACKAGE_BUILD_TARGET_TYPE_CLIENT
        shell_interpreter = self.job_config.executable_interpreter
        if os.path.exists(self.job_config.executable_file_folder):
            source_full_path = os.path.join(self.job_config.executable_file_folder, self.job_config.executable_file)
            server_source_full_path = os.path.join(self.job_config.executable_file_folder,
                                                   self.job_config.server_executable_file)
        else:
            source_full_path = os.path.join(self.job_config.base_dir, self.job_config.executable_file_folder,
                                            self.job_config.executable_file)
            server_source_full_path = os.path.join(self.job_config.base_dir, self.job_config.executable_file_folder,
                                                   self.job_config.server_executable_file)
        source_full_folder = os.path.dirname(source_full_path)
        source_folder = os.path.dirname(self.job_config.executable_file)
        entry_point = os.path.basename(self.job_config.executable_file)
        if os.path.exists(self.job_config.executable_conf_file_folder):
            config_full_path = os.path.join(self.job_config.executable_conf_file_folder,
                                            self.job_config.executable_conf_file)
        else:
            config_full_path = os.path.join(self.job_config.base_dir, self.job_config.executable_conf_file_folder,
                                            self.job_config.executable_conf_file)
        if config_full_path == source_full_path:
            config_full_path = os.path.join(os.path.dirname(config_full_path), "config",
                                            self.job_config.executable_conf_file)
            self.job_config.executable_conf_file_folder = os.path.join(self.job_config.executable_conf_file_folder,
                                                                       "config")
        config_full_folder = os.path.dirname(config_full_path)
        os.makedirs(source_full_folder, exist_ok=True)
        os.makedirs(config_full_folder, exist_ok=True)
        if not os.path.exists(config_full_folder):
            self.job_config.executable_conf_file_folder = os.path.join(Constants.get_fedml_home_dir(),
                                                                       Constants.FEDML_LAUNCH_JOB_TEMP_DIR,
                                                                       self.job_config.executable_conf_file_folder)
            config_full_path = config_full_path = os.path.join(self.job_config.executable_conf_file_folder,
                                                               self.job_config.executable_conf_file)
            config_full_folder = os.path.dirname(config_full_path)
        config_folder = self.job_config.executable_conf_file_folder
        dest_folder = os.path.join(Constants.get_fedml_home_dir(), Constants.FEDML_LAUNCH_JOB_TEMP_DIR)
        bootstrap_full_path = os.path.join(source_full_folder, Constants.BOOTSTRAP_FILE_NAME)
        bootstrap_file = os.path.join(source_full_folder, Constants.BOOTSTRAP_FILE_NAME)
        if platform.system() == Constants.OS_PLATFORM_WINDOWS:
            bootstrap_full_path = bootstrap_full_path.replace('.sh', '.bat')
        os.makedirs(dest_folder, exist_ok=True)

        # Check the paths.
        if not os.path.exists(source_full_path) or self.job_config.using_easy_mode:
            os.makedirs(source_full_folder, exist_ok=True)
            with open(source_full_path, 'w') as source_file_handle:
                if self.job_config.using_easy_mode:
                    source_file_handle.writelines(self.job_config.executable_commands)
                source_file_handle.close()
        if not os.path.exists(server_source_full_path) or self.job_config.using_easy_mode:
            if self.job_config.server_job is not None:
                os.makedirs(source_full_folder, exist_ok=True)
                with open(server_source_full_path, 'w') as server_source_file_handle:
                    if self.job_config.using_easy_mode:
                        server_source_file_handle.writelines(self.job_config.server_job)
                    server_source_file_handle.close()
        if not os.path.exists(config_full_path) or self.job_config.using_easy_mode:
            os.makedirs(config_full_folder, exist_ok=True)
            with open(config_full_path, 'w') as config_file_handle:
                config_file_handle.writelines(
                    ["environment_args:\n", f"  bootstrap: {Constants.BOOTSTRAP_FILE_NAME}\n"])
                config_file_handle.close()

        # Write bootstrap commands into the bootstrap file.
        configs = load_yaml_config(config_full_path)
        configs[Constants.STD_CONFIG_ENV_SECTION][Constants.STD_CONFIG_ENV_SECTION_BOOTSTRAP_KEY] = \
            Constants.BOOTSTRAP_FILE_NAME
        Constants.generate_yaml_doc(configs, config_full_path)
        with open(bootstrap_full_path, 'w') as bootstrap_file_handle:
            bootstrap_file_handle.writelines(self.job_config.bootstrap)
            bootstrap_file_handle.close()
        configs[Constants.LAUNCH_PARAMETER_JOB_YAML_KEY] = self.job_config.job_config_dict

        # Build the client package.
        build_client_package = FedMLLaunchManager.build_job_package(platform_str, client_server_type,
                                                                    source_full_folder,
                                                                    entry_point, config_full_folder, dest_folder, "")
        if build_client_package is None:
            click.echo("Failed to build the application package for the client executable file.")
            exit(-1)

        # Build the server package.
        if self.job_config.server_job is not None:
            client_server_type = Constants.FEDML_PACKAGE_BUILD_TARGET_TYPE_SERVER
            server_entry_point = os.path.basename(self.job_config.server_executable_file)
            build_server_package = FedMLLaunchManager.build_job_package(platform_str, client_server_type,
                                                                        source_full_folder,
                                                                        server_entry_point, config_full_folder,
                                                                        dest_folder, "")
            if build_server_package is None:
                click.echo("Failed to build the application package for the server executable file.")
                exit(-1)
        else:
            build_server_package = None

        # Create and update an application with the built packages.
        FedMLAppManager.get_instance().set_config_version(self.config_version)
        app_updated_result = FedMLAppManager.get_instance().update_app(platform_type,
                                                                       self.job_config.application_name, configs,
                                                                       user_api_key,
                                                                       client_package_file=build_client_package,
                                                                       server_package_file=build_server_package)
        if not app_updated_result:
            click.echo("Failed to upload the application package to MLOps.")
            exit(-1)

        # Start the job with the above application.
        FedMLJobManager.get_instance().set_config_version(self.config_version)
        launch_result = FedMLJobManager.get_instance().start_job(platform_str, self.job_config.project_name,
                                                                 self.job_config.application_name,
                                                                 device_server, device_edges, user_api_key,
                                                                 no_confirmation=no_confirmation)
        if launch_result is not None:
            launch_result.project_name = self.job_config.project_name
            launch_result.application_name = self.job_config.application_name
        # print(f"launch_result = {launch_result}")
        return launch_result

    def start_job(self, platform_type, project_name, application_name,
                  device_server, device_edges,
                  user_api_key, no_confirmation=True, job_id=None):
        launch_result = FedMLJobManager.get_instance().start_job(platform_type, project_name,
                                                                 application_name,
                                                                 device_server, device_edges, user_api_key,
                                                                 no_confirmation=no_confirmation, job_id=job_id)
        if launch_result is not None:
            launch_result.project_name = self.job_config.project_name
            launch_result.application_name = self.job_config.application_name
        return launch_result

    def parse_job_yaml(self, yaml_file):
        self.job_config = FedMLJobConfig(yaml_file)

    @staticmethod
    def build_job_package(platform, client_server_type, source_folder, entry_point,
                          config_folder, dest_folder, ignore, verbose=False):
        if verbose:
            print("Argument for type: " + client_server_type)
            print("Argument for source folder: " + source_folder)
            print("Argument for entry point: " + entry_point)
            print("Argument for config folder: " + config_folder)
            print("Argument for destination package folder: " + dest_folder)
            print("Argument for ignore lists: " + ignore)

        if not platform_is_valid(platform):
            return

        if client_server_type == "client" or client_server_type == "server":
            if verbose:
                print(
                    "Now, you are building the fedml packages which will be used in the MLOps "
                    "platform."
                )
                print(
                    "The packages will be used for client training and server aggregation."
                )
                print(
                    "When the building process is completed, you will find the packages in the directory as follows: "
                    + os.path.join(dest_folder, "dist-packages")
                    + "."
                )
                print(
                    "Then you may upload the packages on the configuration page in the MLOps platform to start the "
                    "federated learning flow."
                )
                print("Building...")
        else:
            if verbose:
                print("You should specify the type argument value as client or server.")
            exit(-1)

        home_dir = expanduser("~")
        mlops_build_path = os.path.join(home_dir, "fedml-mlops-build")
        try:
            shutil.rmtree(mlops_build_path, ignore_errors=True)
        except Exception as e:
            pass

        ignore_list = "{},{}".format(ignore, Constants.FEDML_MLOPS_BUILD_PRE_IGNORE_LIST)
        pip_source_dir = os.path.dirname(__file__)
        pip_build_path = os.path.join(pip_source_dir, "build-package")
        build_dir_ignore = "__pycache__,*.pyc,*.git"
        build_dir_ignore_list = tuple(build_dir_ignore.split(','))
        shutil.copytree(pip_build_path, mlops_build_path,
                        ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*build_dir_ignore_list))

        if client_server_type == "client":
            result = FedMLLaunchManager.build_mlops_package(
                ignore_list,
                source_folder,
                entry_point,
                config_folder,
                dest_folder,
                mlops_build_path,
                "fedml-client",
                "client-package",
                "${FEDSYS.CLIENT_INDEX}",
            )
            if result != 0:
                exit(result)

            build_result_package = os.path.join(dest_folder, "dist-packages", "client-package.zip")
            if verbose:
                print("You have finished all building process. ")
                print(
                    "Now you may use "
                    + build_result_package
                    + " to start your federated "
                      "learning run."
                )

            return build_result_package
        elif client_server_type == "server":
            result = FedMLLaunchManager.build_mlops_package(
                ignore_list,
                source_folder,
                entry_point,
                config_folder,
                dest_folder,
                mlops_build_path,
                "fedml-server",
                "server-package",
                "0",
            )
            if result != 0:
                exit(result)

            build_result_package = os.path.join(dest_folder, "dist-packages", "server-package.zip")
            if verbose:
                print("You have finished all building process. ")
                print(
                    "Now you may use "
                    + os.path.join(dest_folder, "dist-packages", "server-package.zip")
                    + " to start your federated "
                      "learning run."
                )

            return build_result_package

    @staticmethod
    def build_mlops_package(
            ignore,
            source_folder,
            entry_point,
            config_folder,
            dest_folder,
            mlops_build_path,
            mlops_package_parent_dir,
            mlops_package_name,
            rank,
            verbose=False
    ):
        if not os.path.exists(source_folder):
            if verbose:
                print("source folder is not exist: " + source_folder)
            return -1

        if not os.path.exists(os.path.join(source_folder, entry_point)):
            if verbose:
                print(
                    "entry file: "
                    + entry_point
                    + " is not exist in the source folder: "
                    + source_folder
                )
            return -1

        if not os.path.exists(config_folder):
            if verbose:
                print("config folder is not exist: " + source_folder)
            return -1

        mlops_src = source_folder
        mlops_src_entry = entry_point
        mlops_conf = config_folder
        cur_dir = mlops_build_path
        mlops_package_base_dir = os.path.join(
            cur_dir, "mlops-core", mlops_package_parent_dir
        )
        package_dir = os.path.join(mlops_package_base_dir, mlops_package_name)
        fedml_dir = os.path.join(package_dir, "fedml")
        mlops_dest = fedml_dir
        mlops_dest_conf = os.path.join(fedml_dir, "config")
        mlops_pkg_conf = os.path.join(package_dir, "conf", "fedml.yaml")
        mlops_dest_entry = os.path.join("fedml", mlops_src_entry)
        mlops_package_file_name = mlops_package_name + ".zip"
        dist_package_dir = os.path.join(dest_folder, "dist-packages")
        dist_package_file = os.path.join(dist_package_dir, mlops_package_file_name)
        ignore_list = tuple(ignore.split(','))

        shutil.rmtree(mlops_dest_conf, ignore_errors=True)
        shutil.rmtree(mlops_dest, ignore_errors=True)
        try:
            shutil.copytree(mlops_src, mlops_dest, copy_function=shutil.copy,
                            ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*ignore_list))
        except Exception as e:
            pass
        try:
            shutil.copytree(mlops_conf, mlops_dest_conf, copy_function=shutil.copy,
                            ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*ignore_list))
        except Exception as e:
            pass
        try:
            os.remove(os.path.join(mlops_dest_conf, "mqtt_config.yaml"))
            os.remove(os.path.join(mlops_dest_conf, "s3_config.yaml"))
        except Exception as e:
            pass

        mlops_pkg_conf_file = open(mlops_pkg_conf, mode="w")
        mlops_pkg_conf_file.writelines(
            [
                "entry_config: \n",
                "  entry_file: " + mlops_dest_entry + "\n",
                "  conf_file: config/fedml_config.yaml\n",
                "dynamic_args:\n",
                "  rank: " + rank + "\n",
                "  run_id: ${FEDSYS.RUN_ID}\n",
                # "  data_cache_dir: ${FEDSYS.PRIVATE_LOCAL_DATA}\n",
                # "  data_cache_dir: /fedml/fedml-package/fedml/data\n",
                "  mqtt_config_path: /fedml/fedml_config/mqtt_config.yaml\n",
                "  s3_config_path: /fedml/fedml_config/s3_config.yaml\n",
                "  log_file_dir: /fedml/fedml-package/fedml/data\n",
                "  log_server_url: ${FEDSYS.LOG_SERVER_URL}\n",
                "  client_id_list: ${FEDSYS.CLIENT_ID_LIST}\n",
                "  client_objects: ${FEDSYS.CLIENT_OBJECT_LIST}\n",
                "  is_using_local_data: ${FEDSYS.IS_USING_LOCAL_DATA}\n",
                "  synthetic_data_url: ${FEDSYS.SYNTHETIC_DATA_URL}\n",
                "  client_num_in_total: ${FEDSYS.CLIENT_NUM}\n",
            ]
        )
        mlops_pkg_conf_file.flush()
        mlops_pkg_conf_file.close()

        local_mlops_package = os.path.join(mlops_package_base_dir, mlops_package_file_name)
        if os.path.exists(local_mlops_package):
            os.remove(os.path.join(mlops_package_base_dir, mlops_package_file_name))
        mlops_archive_name = os.path.join(mlops_package_base_dir, mlops_package_name)
        shutil.make_archive(
            mlops_archive_name,
            "zip",
            root_dir=mlops_package_base_dir,
            base_dir=mlops_package_name,
        )
        if not os.path.exists(dist_package_dir):
            os.makedirs(dist_package_dir, exist_ok=True)
        if os.path.exists(dist_package_file) and not os.path.isdir(dist_package_file):
            os.remove(dist_package_file)
        mlops_archive_zip_file = mlops_archive_name + ".zip"
        if os.path.exists(mlops_archive_zip_file):
            shutil.move(mlops_archive_zip_file, dist_package_file)

        shutil.rmtree(mlops_build_path, ignore_errors=True)

        return 0

    @staticmethod
    def save_api_key(api_key):
        try:
            os.makedirs(Constants.get_secret_dir(), exist_ok=True)

            with open(Constants.get_launch_secret_file(), 'w') as secret_file_handle:
                secret_file_handle.writelines([api_key])
                secret_file_handle.close()
        except Exception as e:
            pass

    @staticmethod
    def get_api_key():
        try:
            with open(Constants.get_launch_secret_file(), 'r') as secret_file_handle:
                api_key = secret_file_handle.readline()
                secret_file_handle.close()
                return api_key
        except Exception as e:
            return ""


'''
For the Job yaml file, please review the call_gpu.yaml :
'''


class FedMLJobConfig(object):
    def __init__(self, job_yaml_file):
        self.job_config_dict = load_yaml_config(job_yaml_file)
        self.fedml_env = self.job_config_dict.get("fedml_env", {})
        self.project_name = self.fedml_env.get("project_name", None)
        self.base_dir = os.path.dirname(job_yaml_file)
        self.using_easy_mode = True
        self.executable_interpreter = "bash"
        self.executable_file_folder = self.job_config_dict.get("workspace", None)
        self.executable_commands = self.job_config_dict.get("job", "")
        self.bootstrap = self.job_config_dict.get("bootstrap", None)
        self.executable_file = None
        self.server_executable_file = None
        self.executable_conf_option = ""
        self.executable_conf_file_folder = None
        self.executable_conf_file = None
        self.executable_args = None
        self.server_job = self.job_config_dict.get("server_job", None)
        expert_mode = self.job_config_dict.get("expert_mode", None)
        if expert_mode is not None:
            self.using_easy_mode = False
            self.executable_interpreter = expert_mode.get("executable_interpreter", None)
            self.executable_commands = None
            self.bootstrap = expert_mode.get("bootstrap", None)
            self.executable_file_folder = expert_mode.get("executable_file_folder", None)
            self.executable_file = expert_mode.get("executable_file", None)
            self.executable_conf_option = expert_mode.get("executable_conf_option", None)
            self.executable_conf_file_folder = expert_mode.get("executable_conf_file_folder", None)
            self.executable_conf_file = expert_mode.get("executable_conf_file", None)
            self.executable_args = expert_mode.get("executable_args", None)
            self.data_location = expert_mode.get("data_location", None)

        default_example_job_dir_name = Constants.LAUNCH_JOB_DEFAULT_FOLDER_NAME
        default_example_job_dir = os.path.join(self.base_dir, default_example_job_dir_name)
        default_example_job_conf_dir_name = os.path.join(default_example_job_dir_name,
                                                         Constants.LAUNCH_JOB_DEFAULT_CONF_FOLDER_NAME)
        default_example_job_conf_dir = os.path.join(self.base_dir, default_example_job_conf_dir_name)
        if self.executable_file is None or self.executable_file == "":
            if self.executable_file_folder is None:
                self.executable_file_folder = default_example_job_dir_name
            else:
                if not os.path.exists(self.executable_file_folder):
                    self.executable_file_folder = os.path.join(self.base_dir, self.executable_file_folder)
            os.makedirs(self.executable_file_folder, exist_ok=True)
            self.executable_file = Constants.LAUNCH_JOB_DEFAULT_ENTRY_NAME

        self.server_executable_file = Constants.LAUNCH_SERVER_JOB_DEFAULT_ENTRY_NAME

        if self.executable_conf_file is None or self.executable_conf_file == "":
            if self.executable_conf_file_folder is None:
                self.executable_conf_file_folder = default_example_job_conf_dir_name \
                    if not os.path.exists(self.executable_file_folder) else \
                    os.path.join(self.executable_file_folder, Constants.LAUNCH_JOB_DEFAULT_CONF_FOLDER_NAME)
            else:
                if not os.path.exists(self.executable_conf_file_folder):
                    self.executable_conf_file_folder = os.path.join(self.base_dir, self.executable_conf_file_folder)
            os.makedirs(self.executable_conf_file_folder, exist_ok=True)
            self.executable_conf_file = Constants.LAUNCH_JOB_DEFAULT_CONF_NAME
        self.executable_file_folder = str(self.executable_file_folder).replace('\\', os.sep).replace('/', os.sep)
        self.executable_conf_file_folder = str(self.executable_conf_file_folder).replace('\\', os.sep).replace('/',
                                                                                                               os.sep)
        self.executable_file = str(self.executable_file).replace('\\', os.sep).replace('/', os.sep)
        self.executable_conf_file = str(self.executable_conf_file).replace('\\', os.sep).replace('/', os.sep)

        computing_obj = self.job_config_dict.get("computing", {})
        self.minimum_num_gpus = computing_obj.get("minimum_num_gpus", 0)
        self.maximum_cost_per_hour = computing_obj.get("maximum_cost_per_hour", "$0")
        self.application_name = FedMLJobConfig.generate_application_name(self.project_name)

    @staticmethod
    def generate_application_name(project_name):
        return "{}-{}-{}-{}".format(Constants.LAUNCH_APP_NAME_PREFIX,
                                    project_name if project_name is not None else Constants.LAUNCH_PROJECT_NAME_DEFAULT,
                                    MLOpsUtils.get_ntp_time(),
                                    str(uuid.uuid4()))
