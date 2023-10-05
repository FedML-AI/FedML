import os
import platform
import shutil
import uuid
from os.path import expanduser

import click

import fedml
from fedml.computing.scheduler.comm_utils import sys_utils

from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.comm_utils.sys_utils import daemon_ota_upgrade_with_version, \
    check_fedml_is_latest_version
from fedml.computing.scheduler.comm_utils.security_utils import get_api_key, save_api_key
from fedml.computing.scheduler.comm_utils.yaml_utils import load_yaml_config
from fedml.computing.scheduler.comm_utils.platform_utils import platform_is_valid
from prettytable import PrettyTable

from fedml.computing.scheduler.scheduler_entry.constants import Constants
from fedml.computing.scheduler.scheduler_entry.app_manager import FedMLAppManager
from fedml.computing.scheduler.scheduler_entry.job_manager import FedMLJobManager
from fedml.computing.scheduler.scheduler_entry.cluster_manager import FedMLClusterManager
from fedml.computing.scheduler.scheduler_entry.app_manager import FedMLModelUploadResult
from fedml.api.constants import ApiConstants



class FedMLLaunchManager(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, "_instance"):
            orig = super(FedMLLaunchManager, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
            cls._instance.init()
        return cls._instance

    def __init__(self):
        self.config_version = fedml.get_env_version()

    @staticmethod
    def get_instance():
        return FedMLLaunchManager()

    def init(self):
        self.matched_results_map = dict()
        self.platform_type = SchedulerConstants.PLATFORM_TYPE_FALCON
        self.device_server = ""
        self.device_edges = ""

    def launch_job(self, yaml_file, user_api_key, cluster, mlops_platform_type,
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

        # Create and update model card with the job yaml file if the task type is serve.
        model_update_result = None
        if self.job_config.task_type == Constants.JOB_TASK_TYPE_DEPLOY or \
                self.job_config.task_type == Constants.JOB_TASK_TYPE_SERVE:
            if self.job_config.serving_model_name is not None and self.job_config.serving_model_name != "":
                self.job_config.model_app_name = self.job_config.serving_model_name

            models = FedMLAppManager.get_instance().check_model_exists(self.job_config.model_app_name, user_api_key)
            if models is None or len(models.model_list) <= 0:
                if not FedMLAppManager.get_instance().check_model_package(self.job_config.workspace):
                    click.echo(f"Please make sure fedml_model_config.yaml exists in your workspace."
                               f"{self.job_config.workspace}")
                    exit(-1)

                model_update_result = FedMLAppManager.get_instance().update_model(self.job_config.model_app_name,
                                                                                  self.job_config.workspace,
                                                                                  user_api_key)
                if model_update_result is None:
                    click.echo("Failed to upload the model package to MLOps.")
                    exit(-1)

                models = FedMLAppManager.get_instance().check_model_exists(self.job_config.model_app_name, user_api_key)
                if models is None or len(models.model_list) <= 0:
                    click.echo("Failed to upload the model package to MLOps.")
                    exit(-1)

                model_update_result.model_id = models.model_list[0].id
                model_update_result.model_version = models.model_list[0].model_version
                model_update_result.endpoint_name = self.job_config.serving_endpoint_name
            else:
                model_update_result = FedMLModelUploadResult(
                    self.job_config.model_app_name, model_id=models.model_list[0].id,
                    model_version=models.model_list[0].model_version,
                    model_storage_url=self.job_config.serving_model_s3_url,
                    endpoint_name=self.job_config.serving_endpoint_name)

            self.parse_job_yaml(yaml_file, should_use_default_workspace=True)

            # Apply model endpoint id and act as job id
            self.job_config.serving_endpoint_id = FedMLJobManager.get_instance().apply_endpoint_id(
                user_api_key, self.job_config.serving_endpoint_name, model_id=models.model_list[0].id,
                model_name=models.model_list[0].model_name, model_version=models.model_list[0].model_version, )
            if self.job_config.serving_endpoint_id is None:
                click.echo("Failed to apply endpoint for your model.")
                exit(-1)

            model_update_result.endpoint_id = self.job_config.serving_endpoint_id

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
                if model_update_result is not None:
                    random = sys_utils.random1(f"FEDML@{user_api_key}", "FEDML@9999GREAT")
                    config_file_handle.writelines(["serving_args:\n",
                                                   f"  model_id: {model_update_result.model_id}\n",
                                                   f"  model_name: {model_update_result.model_name}\n",
                                                   f"  model_version: {model_update_result.model_version}\n",
                                                   f"  model_storage_url: {model_update_result.model_storage_url}\n",
                                                   f"  endpoint_name: {model_update_result.endpoint_name}\n",
                                                   f"  endpoint_id: {model_update_result.endpoint_id}\n",
                                                   f"  random: {random}\n"])
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
        app_updated_result = FedMLAppManager.get_instance().update_app(
            platform_type, self.job_config.application_name, configs, user_api_key,
            client_package_file=build_client_package, server_package_file=build_server_package,
            workspace=self.job_config.workspace, model_name=self.job_config.serving_model_name,
            model_version=self.job_config.serving_model_version,
            model_url=self.job_config.serving_model_s3_url)
        if not app_updated_result:
            click.echo("Failed to upload the application package to MLOps.")
            exit(-1)

        # Start the job with the above application.
        launch_result = FedMLJobManager.get_instance().start_job(
            platform_str, self.job_config.project_name, self.job_config.application_name,
            device_server, device_edges, user_api_key, cluster=cluster, no_confirmation=no_confirmation,
            model_name=self.job_config.serving_model_name, model_endpoint=self.job_config.serving_endpoint_name,
            job_yaml=self.job_config.job_config_dict, job_type=self.job_config.task_type)
        if launch_result is not None:
            launch_result.inner_id = self.job_config.serving_endpoint_id \
                if self.job_config.task_type == Constants.JOB_TASK_TYPE_DEPLOY or \
                   self.job_config.task_type == Constants.JOB_TASK_TYPE_SERVE else None
            launch_result.project_name = self.job_config.project_name
            launch_result.application_name = self.job_config.application_name
        return launch_result

    def start_job(self, platform_type, project_name, application_name,
                  device_server, device_edges,
                  user_api_key, cluster="", no_confirmation=True, job_id=None, job_type="train"):
        launch_result = FedMLJobManager.get_instance().start_job(platform_type, project_name,
                                                                 application_name,
                                                                 device_server, device_edges, user_api_key, cluster,
                                                                 no_confirmation=no_confirmation, job_id=job_id, job_type=job_type)
        if launch_result is not None:
            launch_result.project_name = self.job_config.project_name
            launch_result.application_name = self.job_config.application_name
        return launch_result

    def parse_job_yaml(self, yaml_file, should_use_default_workspace=False):
        self.job_config = FedMLJobConfig(yaml_file, should_use_default_workspace=should_use_default_workspace)

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

    def check_heartbeat(self, api_key):
        return FedMLJobManager.get_instance().check_heartbeat(api_key)

    def show_resource_type(self):
        return FedMLJobManager.get_instance().show_resource_type()

    def check_api_key(self, api_key=None):
        if api_key is None or api_key == "":
            saved_api_key = get_api_key()
            if saved_api_key is None or saved_api_key == "":
                api_key = click.prompt("FedML® Launch API Key is not set yet, please input your API key")
            else:
                api_key = saved_api_key

        is_valid_heartbeat = FedMLLaunchManager.get_instance().check_heartbeat(api_key)
        if not is_valid_heartbeat:
            return False
        else:
            save_api_key(api_key)
            return True
        
    def fedml_login(self, api_key=None):
        """
        init the launch environment
        :param api_key: API Key from MLOPs
        :param version: dev, test, release
        :return int: error code (0 means successful), str: error message
        """
        api_key_is_valid = self.check_api_key(api_key=api_key)
        if api_key_is_valid:
            return 0, "Login successfully"

        return -1, "Login failed"

    def check_match_result(self, result, yaml_file, prompt=True):
        if result.status == Constants.JOB_START_STATUS_INVALID:
            click.echo(f"\nPlease check your {os.path.basename(yaml_file)} file "
                       f"to make sure the syntax is valid, e.g. "
                       f"whether minimum_num_gpus or maximum_cost_per_hour is valid.")
            return ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED
        elif result.status == Constants.JOB_START_STATUS_BLOCKED:
            click.echo("\nBecause the value of maximum_cost_per_hour is too low,"
                       "we can not find exactly matched machines for your job. \n"
                       "But here we still present machines closest to your expected price as below.")
            return ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED
        elif result.status == Constants.JOB_START_STATUS_QUEUED:
            click.echo("\nNo resource available now, but we can keep your job in the waiting queue.")
            if not prompt:
                return ApiConstants.RESOURCE_MATCHED_STATUS_QUEUED
            if click.confirm("Do you want to join the queue?", abort=False):
                click.echo("You have confirmed to keep your job in the waiting list.")
                return ApiConstants.RESOURCE_MATCHED_STATUS_QUEUED
            else:
                FedMLJobManager.get_instance().stop_job(
                    self.platform_type, get_api_key(), result.job_id)
                return ApiConstants.RESOURCE_MATCHED_STATUS_QUEUE_CANCELED
        elif result.status == Constants.JOB_START_STATUS_BIND_CREDIT_CARD_FIRST:
            click.echo("Please bind your credit card before launching the job.")
            return ApiConstants.RESOURCE_MATCHED_STATUS_BIND_CREDIT_CARD_FIRST
        elif result.status == Constants.JOB_START_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED:
            click.echo("Failed to query credit card binding status. Please try again later.")
            return ApiConstants.RESOURCE_MATCHED_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED

        if result.job_url == "":
            if result.message is not None:
                click.echo(f"Failed to launch the job with response messages: {result.message}")
            else:
                click.echo(f"Failed to launch the job. Please check if the network is available.")
            return ApiConstants.RESOURCE_MATCHED_STATUS_JOB_URL_ERROR

        return ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED

    def show_matched_resource(self, result):
        gpu_matched = getattr(result, "gpu_matched", None)
        if gpu_matched is not None and len(gpu_matched) > 0:
            click.echo(f"\nSearched and matched the following GPU resource for your job:")
            gpu_table = PrettyTable(['Provider', 'Instance', 'vCPU(s)', 'Memory(GB)', 'GPU(s)',
                                     'Region', 'Cost', 'Selected'])
            for gpu_device in gpu_matched:
                gpu_table.add_row([gpu_device.gpu_provider, gpu_device.gpu_instance, gpu_device.cpu_count,
                                   gpu_device.mem_size,
                                   f"{gpu_device.gpu_type}:{gpu_device.gpu_num}",
                                   gpu_device.gpu_region, gpu_device.cost, Constants.CHECK_MARK_STRING])
            print(gpu_table)
            click.echo("")

            click.echo(f"You can also view the matched GPU resource with Web UI at: ")
            click.echo(f"{result.job_url}")

            return gpu_matched

        return None

    # inputs: yaml file
    # return: resource_id, error_code (0 means successful), error_message,
    def api_match_resources(self, yaml_file, cluster="", prompt=True):
        """
        launch a job
        :param prompt:
        :param yaml_file: full path of your job yaml file
        :returns: str: resource id, project_id, int: error code (0 means successful), str: error message
        """
        api_key = get_api_key()

        result = FedMLLaunchManager.get_instance().launch_job(yaml_file, api_key, cluster,
                                                              self.platform_type,
                                                              self.device_server, self.device_edges)
        if result is not None:
            checked_result = self.check_match_result(result, yaml_file, prompt=prompt)
            if checked_result != ApiConstants.RESOURCE_MATCHED_STATUS_MATCHED:
                return result.job_id, result.project_id, ApiConstants.ERROR_CODE[checked_result], checked_result

            gpu_matched = self.show_matched_resource(result)
            if gpu_matched is None:
                return result.job_id, result.project_id, ApiConstants.ERROR_CODE[
                    ApiConstants.RESOURCE_MATCHED_STATUS_NO_RESOURCES], \
                    ApiConstants.RESOURCE_MATCHED_STATUS_NO_RESOURCES

            self.matched_results_map[result.job_id] = result

            return result.job_id, result.project_id, 0, "Successfully"

        return None, None, ApiConstants.ERROR_CODE[ApiConstants.RESOURCE_MATCHED_STATUS_REQUEST_FAILED], \
            ApiConstants.RESOURCE_MATCHED_STATUS_REQUEST_FAILED

    # inputs: yaml file, cluster, resource id
    # return: job_id, error_code (0 means successful), error_message,
    def api_launch_job(self, yaml_file, cluster="", resource_id=None, prompt=True):
        # Check if resource is available
        result = self.matched_results_map.get(resource_id, None) if resource_id is not None else None
        if result is None:
            resource_id, project_id, error_code, error_msg = self.api_match_resources(yaml_file=yaml_file, cluster=cluster, prompt=prompt)
            result = self.matched_results_map.get(resource_id, None) if resource_id is not None else None
            if result is None:
                return resource_id, project_id, error_code, error_msg

        # Confirm to launch job
        if prompt and not click.confirm(f"Are you sure to launch it?", abort=False):
            FedMLJobManager.get_instance().stop_job(
                self.platform_type, get_api_key(), resource_id)
            return result.job_id, result.project_id, ApiConstants.ERROR_CODE[
                ApiConstants.LAUNCH_JOB_STATUS_JOB_CANCELED], \
                ApiConstants.LAUNCH_JOB_STATUS_JOB_CANCELED

        # Get the API key
        api_key = get_api_key()

        # Start the job
        job_id = result.job_id
        ret_job_id = job_id if result.inner_id is None else result.inner_id
        project_id = result.project_id
        cluster_id = result.cluster_id
        gpu_matched = result.gpu_matched
        cluster_confirmed = True
        if not (cluster_id is None or cluster_id == ""):
            cluster_confirmed = FedMLClusterManager.get_instance().confirm_cluster(cluster_id, gpu_matched)
            if not cluster_confirmed:
                return job_id, project_id, cluster_id, ApiConstants.ERROR_CODE[ApiConstants.CLUSTER_CONFIRM_FAILED], \
                    ApiConstants.CLUSTER_CONFIRM_FAILED
            else:
                return job_id, project_id, cluster_id, ApiConstants.ERROR_CODE[ApiConstants.CLUSTER_CONFIRM_SUCCESS], \
                    ApiConstants.CLUSTER_CONFIRM_SUCCESS

        result = FedMLLaunchManager.get_instance().start_job(self.platform_type, result.project_name,
                                                                 result.application_name,
                                                                 self.device_server, self.device_edges, api_key, cluster,
                                                                 no_confirmation=True, job_id=result.job_id, job_type=self.job_config.task_type)
        if result is None:
            return job_id, project_id, ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_REQUEST_FAILED], \
                ApiConstants.LAUNCH_JOB_STATUS_REQUEST_FAILED

        if result.job_url == "":
            if result.message is not None:
                click.echo(f"Failed to launch the job with response messages: {result.message}")
            return result.job_id, project_id, ApiConstants.ERROR_CODE[ApiConstants.LAUNCH_JOB_STATUS_JOB_URL_ERROR], \
                ApiConstants.LAUNCH_JOB_STATUS_JOB_URL_ERROR

        # List the job status
        job_list_obj = FedMLJobManager.get_instance().list_job(self.platform_type, result.project_name,
                                                               None, api_key, job_id=result.job_id)
        if job_list_obj is not None and len(job_list_obj.job_list) > 0:
            click.echo("")
            click.echo("Your launch result is as follows:")
            job_list_table = PrettyTable(['Job Name', 'Job ID', 'Status', 'Created',
                                          'Spend Time(hour)', 'Cost'])
            jobs_count = 0
            for job in job_list_obj.job_list:
                jobs_count += 1
                job_list_table.add_row([job.job_name, job.job_id, job.status, job.started_time,
                                        job.compute_duration, job.cost])
            print(job_list_table)
        else:
            click.echo("")

        # Show the job url
        click.echo("\nYou can track your job running details at this URL:")
        click.echo(f"{result.job_url}")

        # Show querying infos for getting job logs
        click.echo("")
        click.echo(f"For querying the realtime status of your job, please run the following command.")
        click.echo(f"fedml job logs -jid {result.job_id}" +
                   "{}".format(f" -v {self.config_version}"))
        return ret_job_id, project_id, 0, ""


'''
For the Job yaml file, please review the call_gpu.yaml :
'''


class FedMLJobConfig(object):
    def __init__(self, job_yaml_file, should_use_default_workspace=False):
        self.job_config_dict = load_yaml_config(job_yaml_file)
        self.fedml_env = self.job_config_dict.get("fedml_env", {})
        self.project_name = self.fedml_env.get("project_name", None)
        self.base_dir = os.path.dirname(job_yaml_file)
        self.using_easy_mode = True
        self.executable_interpreter = "bash"
        workspace = self.job_config_dict.get("workspace", None)
        self.executable_file_folder = os.path.normpath(
            os.path.join(self.base_dir, workspace)) \
            if not should_use_default_workspace else None
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
                self.executable_file_folder = default_example_job_dir
            else:
                if not os.path.exists(self.executable_file_folder):
                    self.executable_file_folder = os.path.join(self.base_dir, self.executable_file_folder)
            os.makedirs(self.executable_file_folder, exist_ok=True)
            self.executable_file = Constants.LAUNCH_JOB_DEFAULT_ENTRY_NAME

        self.server_executable_file = Constants.LAUNCH_SERVER_JOB_DEFAULT_ENTRY_NAME

        if self.executable_conf_file is None or self.executable_conf_file == "":
            if self.executable_conf_file_folder is None:
                self.executable_conf_file_folder = default_example_job_conf_dir \
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
        self.task_type = self.job_config_dict.get("task_type", None)
        if self.task_type is None:
            self.task_type = self.job_config_dict.get("job_type", Constants.JOB_TASK_TYPE_TRAIN)
        self.framework_type = self.job_config_dict.get("framework_type", Constants.JOB_FRAMEWORK_TYPE_GENERAL)
        self.device_type = computing_obj.get("device_type", Constants.JOB_DEVICE_TYPE_GPU)
        self.resource_type = computing_obj.get("resource_type", "")
        self.workspace = self.executable_file_folder
        serving_args = self.job_config_dict.get("serving_args", {})
        self.serving_model_name = serving_args.get("model_name", None)
        self.serving_model_version = serving_args.get("model_version", "")
        self.serving_model_s3_url = serving_args.get("model_storage_url", "")
        self.serving_endpoint_name = serving_args.get("endpoint_name", None)
        if self.serving_endpoint_name is None or self.serving_endpoint_name == "":
            self.serving_endpoint_name = f"Endpoint-{str(uuid.uuid4())}"
        self.serving_endpoint_id = None

        self.application_name = FedMLJobConfig.generate_application_name(
            self.executable_file_folder if workspace is None or workspace == "" else workspace)

        self.model_app_name = self.serving_model_name \
            if self.serving_model_name is not None and self.serving_model_name != "" else self.application_name

    @staticmethod
    def generate_application_name(workspace):
        return "{}_{}".format(os.path.basename(workspace), Constants.LAUNCH_APP_NAME_PREFIX)
