import os
import platform
import shutil
import uuid
from os.path import expanduser

import fedml
from fedml.computing.scheduler.comm_utils import sys_utils

from fedml.computing.scheduler.comm_utils.sys_utils import upgrade_if_not_latest
from fedml.computing.scheduler.comm_utils.security_utils import get_api_key
from fedml.computing.scheduler.comm_utils.yaml_utils import load_yaml_config
from fedml.computing.scheduler.comm_utils.platform_utils import validate_platform

from fedml.computing.scheduler.scheduler_entry.constants import Constants
from fedml.computing.scheduler.scheduler_entry.app_manager import FedMLAppManager
from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards
from fedml.computing.scheduler.scheduler_entry.app_manager import FedMLModelUploadResult
from fedml.api.modules.utils import build_mlops_package
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants

from fedml.core.common.singleton import Singleton


class FedMLLaunchManager(Singleton):

    def __init__(self):
        self.config_version = fedml.get_env_version()
        self.matched_results_map = dict()
        self.platform_type = SchedulerConstants.PLATFORM_TYPE_FALCON

    @staticmethod
    def get_instance():
        return FedMLLaunchManager()

    def get_matched_result(self, resource_id):
        return self.matched_results_map.get(resource_id, None) if resource_id is not None else None

    def update_matched_result_if_gpu_matched(self, resource_id, result):
        if result is not None:
            gpu_matched = getattr(result, "gpu_matched", None)
            if gpu_matched is not None:
                self.matched_results_map[resource_id] = result

    def prepare_launch(self, yaml_file):
        user_api_key = get_api_key()
        if not os.path.exists(yaml_file):
            raise Exception(f"{yaml_file} can not be found. Please specify the full path of your job yaml file.")

        if os.path.dirname(yaml_file) == "":
            yaml_file = os.path.join(os.getcwd(), yaml_file)

        # Parse the job yaml file and regenerated application name if the job name is not given.
        self._parse_job_yaml(yaml_file)

        # Create and update model card with the job yaml file if the task type is serve.
        model_update_result = self._create_and_update_model_card(yaml_file, user_api_key)

        # Generate source, config and bootstrap related paths.
        fedml_launch_paths = FedMLLaunchPath(self.job_config)

        # Check the paths.
        self._check_paths(fedml_launch_paths, self.job_config, model_update_result, user_api_key)

        # Write bootstrap commands into the bootstrap file.
        app_config = self._write_bootstrap_file(self.job_config, fedml_launch_paths)

        # Build the client package.
        client_package = self._build_client_package(self.platform_type, fedml_launch_paths, self.job_config)

        # Build the server package.
        server_package = self._build_server_package(self.platform_type, fedml_launch_paths, self.job_config)

        return self.job_config, app_config, client_package, server_package

    def post_launch(self, job_config):
        package_dest_folder = os.path.join(Constants.get_fedml_home_dir(), Constants.FEDML_LAUNCH_JOB_TEMP_DIR,
                                           job_config.application_name)
        shutil.rmtree(package_dest_folder)

    def cleanup_launch(self, run_id, inner_id):
        user_api_key = get_api_key()

    def _create_and_update_model_card(self, yaml_file, user_api_key):
        if self.job_config.task_type == Constants.JOB_TASK_TYPE_DEPLOY or \
                self.job_config.task_type == Constants.JOB_TASK_TYPE_SERVE:
            model_app_name = self.job_config.serving_model_name
            if self.job_config.serving_model_name is not None and self.job_config.serving_model_name != "":
                self.job_config.model_app_name = self.job_config.serving_model_name
                model_app_name = self.job_config.model_app_name

            models = FedMLAppManager.get_instance().check_model_exists(self.job_config.model_app_name, user_api_key)
            if models is None or len(models.model_list) <= 0:
                if not FedMLAppManager.get_instance().check_model_package(self.job_config.workspace):
                    raise Exception(f"Please make sure fedml_model_config.yaml exists in your workspace."
                                    f"{self.job_config.workspace}")

                model_update_result = FedMLAppManager.get_instance().update_model(self.job_config.model_app_name,
                                                                                  self.job_config.workspace,
                                                                                  user_api_key)
                if model_update_result is None:
                    raise Exception("Failed to upload the model package to MLOps.")

                models = FedMLAppManager.get_instance().check_model_exists(self.job_config.model_app_name, user_api_key)
                if models is None or len(models.model_list) <= 0:
                    raise Exception("Failed to list the model package from MLOps.")

                model_update_result.model_id = models.model_list[0].id
                model_update_result.model_version = models.model_list[0].latest_model_version \
                    if self.job_config.serving_model_version is None else self.job_config.serving_model_version
                model_update_result.endpoint_name = self.job_config.serving_endpoint_name
            else:
                model_update_result = FedMLAppManager.get_instance().update_model(self.job_config.model_app_name,
                                                                                  self.job_config.workspace,
                                                                                  user_api_key,
                                                                                  is_creating_model=False,
                                                                                  model_object=models.model_list[0])
                if model_update_result is None:
                    raise Exception("Failed to upload the model package to MLOps.")
                models = FedMLAppManager.get_instance().check_model_exists(self.job_config.model_app_name, user_api_key)
                if models is None or len(models.model_list) <= 0:
                    raise Exception("Failed to list the model package from MLOps.")

                model_update_result = FedMLModelUploadResult(
                    self.job_config.model_app_name, model_id=models.model_list[0].id,
                    model_version=models.model_list[0].latest_model_version \
                        if self.job_config.serving_model_version is None else self.job_config.serving_model_version,
                    model_storage_url=self.job_config.serving_model_s3_url,
                    endpoint_name=self.job_config.serving_endpoint_name)

            self._parse_job_yaml(yaml_file, should_use_default_workspace=True)
            self.job_config.model_app_name = model_app_name

            # Apply model endpoint id and act as job id
            applied_endpoint_id = FedMLModelCards.get_instance().apply_endpoint_api(
                user_api_key, self.job_config.serving_endpoint_name, model_id=models.model_list[0].id,
                model_name=models.model_list[0].model_name, model_version=models.model_list[0].latest_model_version,
                endpoint_id=self.job_config.serving_endpoint_id)
            if applied_endpoint_id is None:
                raise Exception("Failed to apply endpoint for your model.")
            if applied_endpoint_id == 0:
                raise Exception("Your endpoint id is occupied by other users.")
            endpoint_detail = FedMLModelCards.get_instance().query_endpoint_detail_api(
                endpoint_id=applied_endpoint_id, user_api_key=user_api_key)
            if endpoint_detail is None:
                raise Exception("Failed to get the endpoint detail.")
            self.job_config.serving_endpoint_id = applied_endpoint_id
            self.job_config.serving_endpoint_name = endpoint_detail.endpoint_name
            self.job_config.serving_model_name = models.model_list[0].model_name
            self.job_config.serving_model_version = models.model_list[0].latest_model_version \
                if self.job_config.serving_model_version is None else self.job_config.serving_model_version
            self.job_config.serving_model_id = models.model_list[0].id

            if self.job_config.job_config_dict.get("serving_args") is None:
                self.job_config.job_config_dict["serving_args"] = dict()
            self.job_config.job_config_dict["serving_args"]["endpoint_id"] = self.job_config.serving_endpoint_id
            self.job_config.job_config_dict["serving_args"]["model_id"] = self.job_config.serving_model_id
            self.job_config.job_config_dict["serving_args"]["model_name"] = self.job_config.serving_model_name
            self.job_config.job_config_dict["serving_args"]["model_version"] = self.job_config.serving_model_version
            self.job_config.job_config_dict["serving_args"]["endpoint_name"] = self.job_config.serving_endpoint_name

            model_update_result = FedMLModelUploadResult(
                self.job_config.model_app_name, model_id=models.model_list[0].id,
                model_version=self.job_config.serving_model_version,
                model_storage_url=self.job_config.serving_model_s3_url,
                endpoint_name=self.job_config.serving_endpoint_name)
            model_update_result.endpoint_id = self.job_config.serving_endpoint_id
            return model_update_result

    def _parse_job_yaml(self, yaml_file, should_use_default_workspace=False):
        self.job_config = FedMLJobConfig(yaml_file, should_use_default_workspace=should_use_default_workspace)

    @staticmethod
    def _write_bootstrap_file(job_config, fedml_launch_paths):
        configs = load_yaml_config(fedml_launch_paths.config_launch_full_path)
        if os.path.exists(fedml_launch_paths.bootstrap_full_path):
            with open(fedml_launch_paths.bootstrap_full_path, 'r') as bootstrap_file_handle:
                bootstrap_lines = bootstrap_file_handle.readlines()
                job_config.bootstrap += "".join(bootstrap_lines)
                bootstrap_file_handle.close()
        fedml_launch_paths.add_tmp_boostrap_file(os.path.join(job_config.tmp_dir,
                                                              os.path.basename(fedml_launch_paths.bootstrap_full_path)))
        if os.path.exists(fedml_launch_paths.bootstrap_full_path):
            shutil.copyfile(fedml_launch_paths.bootstrap_full_path, fedml_launch_paths.tmp_bootstrap_file)
        with open(fedml_launch_paths.bootstrap_full_path, 'w') as bootstrap_file_handle:
            if job_config.bootstrap_on_posix is not None:
                bootstrap_file_handle.writelines(job_config.bootstrap_on_posix)
            else:
                bootstrap_file_handle.write("\n")
            bootstrap_file_handle.close()

        fedml_launch_paths.tmp_bootstrap_file_on_windows = os.path.join(
            job_config.tmp_dir, os.path.basename(fedml_launch_paths.bootstrap_full_path_on_windows))
        if os.path.exists(fedml_launch_paths.bootstrap_full_path_on_windows):
            shutil.copyfile(
                fedml_launch_paths.bootstrap_full_path_on_windows, fedml_launch_paths.tmp_bootstrap_file_on_windows)
        with open(fedml_launch_paths.bootstrap_full_path_on_windows, 'w') as bootstrap_file_handle:
            if job_config.bootstrap_on_windows is not None:
                bootstrap_file_handle.writelines(job_config.bootstrap_on_windows)
            else:
                bootstrap_file_handle.write("\n")
            bootstrap_file_handle.close()
        configs[Constants.LAUNCH_PARAMETER_JOB_YAML_KEY] = job_config.job_config_dict

        return configs

    @staticmethod
    def _check_paths(fedml_launch_paths, job_config, model_update_result, user_api_key):
        if not os.path.exists(fedml_launch_paths.source_full_path) or job_config.using_easy_mode:
            os.makedirs(fedml_launch_paths.source_full_folder, exist_ok=True)
            with open(fedml_launch_paths.source_full_path, 'w') as source_file_handle:
                if job_config.using_easy_mode and job_config.commands_on_posix is not None:
                    source_file_handle.writelines(job_config.commands_on_posix)
                source_file_handle.close()
        if not os.path.exists(fedml_launch_paths.server_source_full_path) or job_config.using_easy_mode:
            if job_config.server_job is not None:
                os.makedirs(fedml_launch_paths.source_full_folder, exist_ok=True)
                with open(fedml_launch_paths.server_source_full_path, 'w') as server_source_file_handle:
                    if job_config.using_easy_mode and job_config.server_job_on_posix is not None:
                        server_source_file_handle.writelines(job_config.server_job_on_posix)
                    server_source_file_handle.close()

        # Generate the source script on Windows
        if not os.path.exists(fedml_launch_paths.source_full_path_on_windows) or job_config.using_easy_mode:
            os.makedirs(fedml_launch_paths.source_full_folder, exist_ok=True)
            with open(fedml_launch_paths.source_full_path_on_windows, 'w') as source_file_handle:
                if job_config.using_easy_mode and job_config.commands_on_windows is not None:
                    source_file_handle.writelines(job_config.commands_on_windows)
                source_file_handle.close()
        if not os.path.exists(fedml_launch_paths.server_source_full_path_on_windows) or job_config.using_easy_mode:
            if job_config.server_job is not None:
                os.makedirs(fedml_launch_paths.source_full_folder, exist_ok=True)
                with open(fedml_launch_paths.server_source_full_path_on_windows, 'w') as server_source_file_handle:
                    if job_config.using_easy_mode and job_config.server_job_on_windows is not None:
                        server_source_file_handle.writelines(job_config.server_job_on_windows)
                    server_source_file_handle.close()

        if job_config.using_easy_mode:
            config_dict = load_yaml_config(fedml_launch_paths.config_full_path) if os.path.exists(
                fedml_launch_paths.config_full_path) else dict()
            if config_dict.get("environment_args", None) is None:
                config_dict["environment_args"] = dict()
            if config_dict["environment_args"].get("bootstrap", None) is None:
                config_dict["environment_args"]["bootstrap"] = Constants.BOOTSTRAP_FILE_NAME
            else:
                bootstrap_file = config_dict["environment_args"]["bootstrap"]
                bootstrap_full_path = os.path.join(fedml_launch_paths.source_full_folder, bootstrap_file)
            for config_name, config_value in job_config.job_config_dict.items():
                if config_name not in Constants.JOB_YAML_RESERVED_CONFIG_KEY_WORDS:
                    config_dict[config_name] = config_value
            if model_update_result is not None:
                random = sys_utils.random1(f"FEDML@{user_api_key}", "FEDML@9999GREAT")
                config_dict["serving_args"] = dict()
                config_dict["serving_args"]["model_id"] = model_update_result.model_id
                config_dict["serving_args"]["model_name"] = model_update_result.model_name
                config_dict["serving_args"]["model_version"] = model_update_result.model_version
                config_dict["serving_args"]["model_storage_url"] = model_update_result.model_storage_url
                config_dict["serving_args"]["endpoint_name"] = model_update_result.endpoint_name
                config_dict["serving_args"]["endpoint_id"] = model_update_result.endpoint_id
                config_dict["serving_args"]["random"] = random
            Constants.generate_yaml_doc(config_dict, fedml_launch_paths.config_launch_full_path)

    @staticmethod
    def _build_client_package(platform_type, fedml_launch_paths, job_config):
        client_server_type = Constants.FEDML_PACKAGE_BUILD_TARGET_TYPE_CLIENT
        build_client_package = FedMLLaunchManager._build_job_package(platform_type, client_server_type,
                                                                     fedml_launch_paths.source_full_folder,
                                                                     fedml_launch_paths.entry_point,
                                                                     fedml_launch_paths.config_launch_full_folder,
                                                                     fedml_launch_paths.dest_folder,
                                                                     job_config.ignore_list_str)
        if os.path.exists(fedml_launch_paths.tmp_bootstrap_file):
            shutil.copyfile(fedml_launch_paths.tmp_bootstrap_file, fedml_launch_paths.bootstrap_full_path)
        if os.path.exists(fedml_launch_paths.tmp_bootstrap_file_on_windows):
            shutil.copyfile(
                fedml_launch_paths.tmp_bootstrap_file_on_windows,
                fedml_launch_paths.bootstrap_full_path_on_windows)
        if build_client_package is None:
            shutil.rmtree(fedml_launch_paths.dest_folder, ignore_errors=True)
            raise Exception("Failed to build the application package for the client executable file.")
        return build_client_package

    @staticmethod
    def _build_server_package(platform_type, fedml_launch_paths, job_config):
        if job_config.server_job is not None:
            client_server_type = Constants.FEDML_PACKAGE_BUILD_TARGET_TYPE_SERVER
            server_entry_point = os.path.basename(job_config.server_executable_file)
            build_server_package = FedMLLaunchManager._build_job_package(platform_type, client_server_type,
                                                                         fedml_launch_paths.source_full_folder,
                                                                         server_entry_point,
                                                                         fedml_launch_paths.config_launch_full_folder,
                                                                         fedml_launch_paths.dest_folder,
                                                                         job_config.ignore_list_str)
            job_config.cleanup_temp_files()
            if build_server_package is None:
                raise Exception("Failed to build the application package for the server executable file.")
        else:
            build_server_package = None
            job_config.cleanup_temp_files()
        return build_server_package

    @staticmethod
    def _cleanup_build_tmp_path(build_path):
        try:
            shutil.rmtree(build_path, ignore_errors=True)
        except Exception:
            pass

    @staticmethod
    def _build_job_package(platform_type, client_server_type, source_folder, entry_point,
                           config_folder, dest_folder, ignore, verbose=False):

        if verbose:
            print("Argument for type: " + client_server_type)
            print("Argument for source folder: " + source_folder)
            print("Argument for entry point: " + entry_point)
            print("Argument for config folder: " + config_folder)
            print("Argument for destination package folder: " + dest_folder)
            print("Argument for ignore lists: " + ignore)

        validate_platform(platform_type)

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
            raise Exception("You should specify the type argument value as client or server.")

        home_dir = expanduser("~")
        mlops_build_path = os.path.join(home_dir, ".fedml", "fedml-mlops-build", str(uuid.uuid4()))
        FedMLLaunchManager._cleanup_build_tmp_path(mlops_build_path)

        ignore_list = "{},{}".format(ignore, Constants.FEDML_MLOPS_BUILD_PRE_IGNORE_LIST)
        pip_source_dir = os.path.dirname(__file__)
        pip_build_path = os.path.join(pip_source_dir, "build-package")
        build_dir_ignore = "__pycache__,*.pyc,*.git"
        build_dir_ignore_list = tuple(build_dir_ignore.split(','))
        shutil.copytree(pip_build_path, mlops_build_path,
                        ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*build_dir_ignore_list))

        if client_server_type == "client":
            result = build_mlops_package(
                ignore_list,
                source_folder,
                entry_point,
                config_folder,
                dest_folder,
                mlops_build_path,
                "fedml-client",
                "client-package",
                "${FEDSYS.CLIENT_INDEX}",
                package_type=SchedulerConstants.JOB_PACKAGE_TYPE_LAUNCH
            )
            FedMLLaunchManager._cleanup_build_tmp_path(mlops_build_path)
            if result != 0:
                raise Exception(f"Failed to build package, result {result}")

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
            result = build_mlops_package(
                ignore_list,
                source_folder,
                entry_point,
                config_folder,
                dest_folder,
                mlops_build_path,
                "fedml-server",
                "server-package",
                "0",
                package_type=SchedulerConstants.JOB_PACKAGE_TYPE_LAUNCH
            )
            FedMLLaunchManager._cleanup_build_tmp_path(mlops_build_path)
            if result != 0:
                raise Exception(f"Failed to build package, result {result}")

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


class FedMLJobConfig(object):
    def __init__(self, job_yaml_file, should_use_default_workspace=False):
        self.job_config_dict = load_yaml_config(job_yaml_file)
        self.fedml_env = self.job_config_dict.get("fedml_env", {})
        self.project_name = self.fedml_env.get("project_name", None)
        self.federate_project_name = self.fedml_env.get("federate_project_name", None)
        self.base_dir = os.path.dirname(job_yaml_file)
        self.using_easy_mode = True
        self.executable_interpreter = "bash"
        workspace = self.job_config_dict.get("workspace", None)
        random_workspace = str(uuid.uuid4())
        self.executable_file_folder = os.path.normpath(
            os.path.join(self.base_dir,
                         workspace if workspace is not None and workspace != "" else random_workspace)) \
            if not should_use_default_workspace else None
        self.executable_commands = self.job_config_dict.get("job", "")
        if isinstance(self.executable_commands, dict):
            self.commands_on_windows = self.executable_commands.get("run_on_windows", "")
            self.commands_on_posix = self.executable_commands.get("run_on_posix", "")
        else:
            self.commands_on_windows = None
            self.commands_on_posix = self.executable_commands
        self.bootstrap = self.job_config_dict.get("bootstrap", None)
        if self.bootstrap is not None and isinstance(self.bootstrap, dict):
            self.bootstrap_on_windows = self.bootstrap.get("run_on_windows", None)
            self.bootstrap_on_posix = self.bootstrap.get("run_on_posix", None)
        else:
            self.bootstrap_on_windows = None
            self.bootstrap_on_posix = self.bootstrap
        self.executable_file = None
        self.server_executable_file = None
        self.executable_conf_option = ""
        self.executable_conf_file_folder = None
        self.executable_conf_file = None
        self.executable_args = None
        self.server_job = self.job_config_dict.get("server_job", None)
        if isinstance(self.server_job, dict):
            self.server_job_on_windows = self.server_job.get("run_on_windows", "")
            self.server_job_on_posix = self.server_job.get("run_on_posix", "")
        else:
            self.server_job_on_windows = None
            self.server_job_on_posix = self.server_job
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
        self.tmp_dir = Constants.get_temp_dir()
        default_example_job_dir = os.path.join(self.tmp_dir, default_example_job_dir_name)
        default_example_job_conf_dir_name = os.path.join(default_example_job_dir_name,
                                                         Constants.LAUNCH_JOB_DEFAULT_CONF_FOLDER_NAME)
        default_example_job_conf_dir = os.path.join(self.tmp_dir, default_example_job_conf_dir_name)
        if self.executable_file is None or self.executable_file == "":
            if self.executable_file_folder is None:
                self.executable_file_folder = default_example_job_dir
            else:
                if not os.path.exists(self.executable_file_folder):
                    self.executable_file_folder = default_example_job_dir
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
        self.task_subtype = self.job_config_dict.get("job_subtype", Constants.JOB_TASK_SUBTYPE_TRAIN_GENERAL_TRAINING)
        self.framework_type = self.job_config_dict.get("framework_type", Constants.JOB_FRAMEWORK_TYPE_GENERAL)
        self.device_type = computing_obj.get("device_type", Constants.JOB_DEVICE_TYPE_GPU)
        self.resource_type = computing_obj.get("resource_type", "")
        self.workspace = self.executable_file_folder
        serving_args = self.job_config_dict.get("serving_args", {})
        self.serving_model_name = serving_args.get("model_name", None)
        self.serving_model_id = serving_args.get("model_id", None)
        self.serving_model_version = serving_args.get("model_version", None)
        self.serving_model_s3_url = serving_args.get("model_storage_url", "")
        self.serving_endpoint_name = serving_args.get("endpoint_name", None)
        if self.serving_endpoint_name is None or self.serving_endpoint_name == "":
            self.serving_endpoint_name = f"Endpoint-{str(uuid.uuid4())}"
        self.serving_endpoint_id = serving_args.get("endpoint_id", None)

        job_args = self.job_config_dict.get("job_args", {})
        self.job_id = job_args.get("job_id", None)
        self.config_id = job_args.get("config_id", None)
        self.job_name = self.job_config_dict.get("job_name", None)

        self.application_name = FedMLJobConfig._generate_application_name(
            random_workspace if self.workspace.startswith(self.tmp_dir) else self.workspace)
        self.application_name = self.job_name if self.job_name is not None else self.application_name

        self.model_app_name = self.serving_model_name \
            if self.serving_model_name is not None and self.serving_model_name != "" else self.application_name

        self.gitignore_file = os.path.join(
            self.base_dir, workspace if workspace is not None and workspace != "" else random_workspace, ".gitignore")
        self.ignore_list_str = Constants.FEDML_MLOPS_BUILD_PRE_IGNORE_LIST
        self.read_gitignore_file()

    @staticmethod
    def _generate_application_name(workspace):
        app_name = os.path.basename(workspace) if workspace is not None and workspace != "" else str(uuid.uuid4())
        return "{}_{}".format(app_name, Constants.LAUNCH_APP_NAME_SUFFIX)

    def cleanup_temp_files(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        conf_folder = os.path.join(os.path.dirname(self.executable_conf_file_folder),
                                   Constants.LAUNCH_JOB_LAUNCH_CONF_FOLDER_NAME)
        shutil.rmtree(conf_folder, ignore_errors=True)

        files_to_remove = []
        source_full_path = os.path.join(self.executable_file_folder, self.executable_file)
        if os.path.exists(source_full_path):
            boostrap_path = os.path.join(self.executable_file_folder, Constants.BOOTSTRAP_FILE_NAME)
            files_to_remove.extend([source_full_path, boostrap_path])

        server_source_full_path = os.path.join(self.executable_file_folder, self.server_executable_file)
        files_to_remove.append(server_source_full_path)

        source_full_path_to_base = os.path.join(self.base_dir, self.executable_file_folder, self.executable_file)
        if os.path.exists(source_full_path_to_base):
            boostrap_path = os.path.join(self.base_dir, self.executable_file_folder, Constants.BOOTSTRAP_FILE_NAME)
            files_to_remove.extend([source_full_path_to_base, boostrap_path])

        server_source_full_path_to_base = os.path.join(self.base_dir, self.executable_file_folder,
                                                       self.server_executable_file)
        if os.path.exists(server_source_full_path_to_base):
            files_to_remove.append(server_source_full_path_to_base)

        sys_utils.remove_files(files_to_remove)
        sys_utils.convert_and_remove_bat_files(files_to_remove)


    def read_gitignore_file(self):
        try:
            ignore_list = list()
            with open(self.gitignore_file, "r") as ignore_file_handle:
                while True:
                    ignore_line = ignore_file_handle.readline()
                    if not ignore_line:
                        break
                    ignore_line = ignore_line.replace('\n', '')
                    if ignore_line.startswith("#") or len(ignore_line.lstrip(' ').rstrip(' ')) == 0:
                        continue
                    ignore_list.append(ignore_line)

                if len(ignore_list) > 0:
                    self.ignore_list_str = ','.join(ignore_list)
                    self.ignore_list_str = self.ignore_list_str.replace("\n", "")
                ignore_file_handle.close()
        except Exception as e:
            pass


class FedMLLaunchPath(object):
    def __init__(self, job_config: FedMLJobConfig):
        self.tmp_bootstrap_file = None
        if os.path.exists(job_config.executable_file_folder):
            self.source_full_path = os.path.join(job_config.executable_file_folder, job_config.executable_file)
            self.server_source_full_path = os.path.join(job_config.executable_file_folder,
                                                        job_config.server_executable_file)
        else:
            self.source_full_path = os.path.join(job_config.base_dir, job_config.executable_file_folder,
                                                 job_config.executable_file)
            self.server_source_full_path = os.path.join(job_config.base_dir, job_config.executable_file_folder,
                                                        job_config.server_executable_file)

        self.source_full_path_on_windows = os.path.join(
            os.path.dirname(self.source_full_path), os.path.basename(self.source_full_path).rstrip(".sh") + ".bat")
        self.server_source_full_path_on_windows = os.path.join(
            os.path.dirname(self.server_source_full_path),
            os.path.basename(self.server_source_full_path).rstrip(".sh") + ".bat")

        self.source_full_folder = os.path.dirname(self.source_full_path)
        self.source_folder = os.path.dirname(job_config.executable_file)
        self.entry_point = os.path.basename(job_config.executable_file)
        if os.path.exists(job_config.executable_conf_file_folder):
            self.config_full_path = os.path.join(job_config.executable_conf_file_folder,
                                                 job_config.executable_conf_file)
        else:
            self.config_full_path = os.path.join(job_config.base_dir, job_config.executable_conf_file_folder,
                                                 job_config.executable_conf_file)
        if self.config_full_path == self.source_full_path:
            self.config_full_path = os.path.join(os.path.dirname(self.config_full_path), "config",
                                                 job_config.executable_conf_file)
            job_config.executable_conf_file_folder = os.path.join(job_config.executable_conf_file_folder,
                                                                  "config")
        self.config_full_folder = os.path.dirname(self.config_full_path)
        self.config_launch_full_path = os.path.join(os.path.dirname(os.path.dirname(self.config_full_path)),
                                                    Constants.LAUNCH_JOB_LAUNCH_CONF_FOLDER_NAME,
                                                    Constants.LAUNCH_JOB_DEFAULT_CONF_NAME)
        self.config_launch_full_folder = os.path.dirname(self.config_launch_full_path)
        os.makedirs(self.config_launch_full_folder, exist_ok=True)
        os.makedirs(self.source_full_folder, exist_ok=True)
        if not os.path.exists(self.config_full_folder):
            job_config.executable_conf_file_folder = os.path.join(Constants.get_fedml_home_dir(),
                                                                  Constants.FEDML_LAUNCH_JOB_TEMP_DIR,
                                                                  job_config.executable_conf_file_folder)
            self.config_full_path = os.path.join(job_config.executable_conf_file_folder,
                                                 job_config.executable_conf_file)
            self.config_full_folder = os.path.dirname(self.config_full_path)
        self.config_folder = job_config.executable_conf_file_folder
        self.dest_folder = os.path.join(Constants.get_fedml_home_dir(), Constants.FEDML_LAUNCH_JOB_TEMP_DIR,
                                        job_config.application_name)
        self.bootstrap_full_path = os.path.join(self.source_full_folder, Constants.BOOTSTRAP_FILE_NAME)
        self.bootstrap_full_path_on_windows = os.path.join(
            os.path.dirname(self.bootstrap_full_path),
            os.path.basename(self.bootstrap_full_path).rstrip(".sh") + ".bat")

        os.makedirs(self.dest_folder, exist_ok=True)

    def add_tmp_boostrap_file(self, path: str):
        self.tmp_bootstrap_file = path
