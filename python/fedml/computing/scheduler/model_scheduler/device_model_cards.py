import argparse
import json
import os
import shutil
import time
import uuid
import yaml

import fedml
import requests
from fedml.core.distributed.communication.mqtt.mqtt_manager import MqttManager

from fedml.core.distributed.communication.s3.remote_storage import S3Storage

from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.model_scheduler.modelops_configs import ModelOpsConfigs
from fedml.computing.scheduler.model_scheduler.device_model_deployment import get_model_info
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from fedml.computing.scheduler.model_scheduler.device_model_object import FedMLModelList, FedMLEndpointDetail
from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants
from fedml.computing.scheduler.comm_utils.security_utils import get_api_key


class FedMLModelCards(Singleton):

    def __init__(self):
        self.current_model_name = None
        self.local_deployment_end_point_id = None
        self.should_end_local_deployment = False
        self.local_deployment_mqtt_mgr = None
        self.config_version = fedml.get_env_version()

    @staticmethod
    def get_instance():
        return FedMLModelCards()

    def serve_model_on_premise(self, model_name, endpoint_name, master_device_ids,
                               worker_device_ids, use_remote, endpoint_id):
        print(f"Use remote: {use_remote}")
        # Check api key
        user_api_key = get_api_key()
        if user_api_key == "":
            print('''
            Please use one of the ways below to login first:
            (1) CLI: `fedml login $api_key`
            (2) API: fedml.api.fedml_login(api_key=$api_key)
            ''')
            return False

        # Concat target devices
        target_devices = self.concat_device_ids(master_device_ids, worker_device_ids)
        if len(target_devices) == 0:
            print("[Error] Please specify both the master device id and worker device ids")
            return False

        # Deploy remote model card
        device_type = "md.on_premise_device"
        additional_params_dict = {}
        use_local_deployment = False

        if use_remote:
            if not self.deploy_model(model_name, device_type, target_devices, "", user_api_key,
                                     additional_params_dict, use_local_deployment, endpoint_id=endpoint_id):
                print("Failed to deploy model")
                return False
            return True

        # Deploy local model card (Build + Push + Deploy)
        local_model_folder = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(local_model_folder):
            print("[Error] Model {} doesn't exist. Please Create it First...".format(model_name))
            return False
        else:
            # Exist a local model folder, recreate the model
            if not self.recreate_model(model_name):
                print("[Error] Failed to recreate model {}.".format(model_name))

        config_file_path = os.path.join(local_model_folder, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE)
        params_dict = self.parse_config_yaml(config_file_path)
        additional_params_dict = params_dict.get("default_params_dict", {})
        use_local_deployment = params_dict.get("default_use_local", False)
        device_type = params_dict.get("device_type", "md.on_premise_device")

        self.build_model(model_name)

        self.push_model(model_name, "", user_api_key)

        if not self.deploy_model(model_name, device_type, target_devices, "", user_api_key,
                                 additional_params_dict, use_local_deployment, endpoint_name=endpoint_name,
                                 endpoint_id=endpoint_id):
            print("Failed to deploy model")
            return False
        return True

    def concat_device_ids(self, master_device_ids, worker_device_ids) -> list:
        if master_device_ids is None or worker_device_ids is None:
            print("Please specify the master device id and worker device ids in the config file.")
            return []
        if not (type(worker_device_ids) in [str, list, int] and type(master_device_ids) in [str, int, list]):
            print('''The format of worker_device_ids is wrong, 
                  please use formate like 1,2,3 or 1,
                  E.g. export FEDML_MODEL_SERVE_WORKER_DEVICE_IDS=1,2,3
                  ''')
            return []

        if type(worker_device_ids) is not list:
            if type(worker_device_ids) is int:
                worker_device_ids = [worker_device_ids]
            else:
                worker_device_ids = worker_device_ids.split(",")
        if type(master_device_ids) is not list:
            master_device_ids = [str(master_device_ids)]
        devices = master_device_ids + worker_device_ids
        return devices

    def parse_config_yaml(self, yaml_file):
        with open(yaml_file, 'r') as f:
            launch_params = yaml.safe_load(f)
        return launch_params

    def copy_config_yaml_to_src_folder(self, src_folder, yaml_file):
        shutil.copy(yaml_file, os.path.join(src_folder, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE))
        with open(os.path.join(src_folder, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE), 'r') as f:
            launch_params = yaml.safe_load(f)
            launch_params["workspace"] = "./"  # Since it is inside the src folder, the workspace is "./"
        with open(os.path.join(src_folder, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE), 'w') as f:
            yaml.dump(launch_params, f, sort_keys=False)
        return True

    def create_model_use_config(self, model_name, config_file) -> bool:
        """
        config_file: the config file path, could be a relative path or absolute path
        workspace: the workspace of the model, could be a relative path or absolute path
        """
        if config_file is None or config_file == "":
            print("[Error] Please specify your config file using --config_file or -cf.")
            return False

        if not os.path.isabs(config_file):
            # Current working directory + config_file if config_file is a relative path
            config_file = os.path.normpath(os.path.join(os.getcwd(), config_file))

        if not os.path.exists(config_file):
            print("[Error] The config file {} doesn't exist.".format(config_file))
            return False

        model_config = self.parse_config_yaml(config_file)
        if "workspace" not in model_config:
            print("Please specify the workspace in the config file.")
            return False

        if not os.path.isabs(model_config["workspace"]):
            # Use config_file_path + workspace if workspace is a relative path
            base_path = os.path.dirname(config_file)  # Avoid scene like: ./src
            workspace_abs_path = os.path.join(base_path, model_config["workspace"])
        else:
            workspace_abs_path = model_config["workspace"]
        workspace_abs_path = os.path.normpath(workspace_abs_path)

        dst_model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        real_dst_model_dir = os.path.realpath(dst_model_dir)
        real_model_root_dir = os.path.realpath(ClientConstants.get_model_dir())
        if not real_dst_model_dir.startswith(real_model_root_dir):   # Avoid deleting parent folders
            print(f"[Error] The destination folder {real_dst_model_dir} is the parent folder of the "
                  f"local model card directory {real_model_root_dir}, "
                  f"please do not include any \"../\" in your model name.")
            return False
        if os.path.exists(dst_model_dir):
            shutil.rmtree(dst_model_dir, ignore_errors=True)

        if self.add_model_files(model_name, workspace_abs_path):
            model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
            if self.copy_config_yaml_to_src_folder(model_dir, config_file):
                # Record the current yaml location and write it to a file in the model dir,
                # So that we can auto-recreate the model next time.
                abs_path_to_config_file = os.path.abspath(config_file)
                with open(os.path.join(model_dir, ClientConstants.ORIGINAL_YAML_FILE_LOCATION), 'w') as f:
                    f.write(abs_path_to_config_file)
                return True
            else:
                print(f"Failed to add your config file {config_file} to the model {model_name}.")
                return False
        else:
            print(f"Failed to add your workspace {workspace_abs_path} to the model {model_name}.")
            return False

    def recreate_model(self, model_name):
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            print(f"[Error] The model {model_name} doesn't exist. Please create it first.")
            return False

        if not os.path.exists(os.path.join(model_dir, ClientConstants.ORIGINAL_YAML_FILE_LOCATION)):
            print(f"The model {model_name} doesn't created by a yaml file, cannot recreate it.")
            return False

        with open(os.path.join(model_dir, ClientConstants.ORIGINAL_YAML_FILE_LOCATION), 'r') as f:
            original_config_file_path = f.read()

        if not os.path.exists(original_config_file_path) or not os.path.isfile(original_config_file_path) \
                or not os.path.isabs(original_config_file_path):
            print(f"The original config file {original_config_file_path} doesn't exist, cannot recreate the model.")
            return False

        # tmp rename the original model cards to avoid conflict
        tmp_model_name = model_name + "_tmp_" + str(int(time.time()))
        shutil.move(model_dir, os.path.join(ClientConstants.get_model_dir(), tmp_model_name))

        if self.create_model_use_config(model_name, original_config_file_path):
            shutil.rmtree(os.path.join(ClientConstants.get_model_dir(), tmp_model_name))
            return True
        else:
            # Rollback
            shutil.move(os.path.join(ClientConstants.get_model_dir(), tmp_model_name),
                        os.path.join(ClientConstants.get_model_dir(), model_name))
            return False

    def create_model(self, model_name):
        self.current_model_name = model_name
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        return True

    def delete_model(self, model_name):
        if model_name == "*":
            shutil.rmtree(ClientConstants.get_model_dir(), ignore_errors=True)
            return True

        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
        else:
            print(f"[Error] The model {model_name} doesn't exist.")
            return False
        return True

    def add_model_files(self, model_name, file_path):
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            self.create_model(model_name)

        if not os.path.exists(model_dir):
            return False

        if not os.path.exists(file_path):
            print(f"[Error] The file {file_path} doesn't exist.")
            return False

        if os.path.isdir(file_path):
            file_ignore = "__pycache__,*.pyc,*.git"
            file_ignore_list = tuple(file_ignore.split(','))
            file_list = os.listdir(file_path)
            for file_item in file_list:
                file_full_path = os.path.join(file_path, file_item)
                if os.path.isdir(file_full_path):
                    dst_dir = os.path.join(model_dir, file_item)
                    if os.path.exists(dst_dir):  # avoid using shutil.copytree(dirs_exist_ok=True)
                        shutil.rmtree(dst_dir)
                    shutil.copytree(file_full_path, dst_dir,
                                    copy_function=shutil.copy,
                                    ignore_dangling_symlinks=True,
                                    ignore=shutil.ignore_patterns(*file_ignore_list),
                                    )
                    if not os.path.exists(dst_dir):
                        print("Directory {} can't be added into the model.".format(file_full_path))
                        return False
                else:
                    file_ignore = "__pycache__,.pyc,.git"
                    src_file_name = os.path.basename(file_full_path)
                    _, src_file_extension = os.path.splitext(file_full_path)
                    dst_file = os.path.join(model_dir, src_file_name)
                    try:
                        file_ignore.split(',').index(src_file_extension)
                    except ValueError as e:
                        shutil.copyfile(file_full_path, dst_file)
                        if not os.path.exists(dst_file):
                            print("File {} can't be added into the model.".format(file_full_path))
                            return False
        else:
            src_file_name = os.path.basename(file_path)
            dst_file = os.path.join(model_dir, src_file_name)
            shutil.copyfile(file_path, dst_file)
            if not os.path.exists(dst_file):
                return False

        return True

    def remove_model_files(self, model_name, file_name):
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            return False

        dst_file = os.path.join(model_dir, file_name)
        os.remove(dst_file)
        if os.path.exists(dst_file):
            return False

        return True

    def list_models(self, model_name, api_key=None):
        if api_key is None:
            model_home_dir = ClientConstants.get_model_dir()
            if not os.path.exists(model_home_dir):
                return []

            models = os.listdir(model_home_dir)
            if model_name == "*":
                models = [model for model in models if not model.startswith('.DS_Store')]
                return models
            else:
                for model in models:
                    if model == model_name:
                        model_dir = os.path.join(model_home_dir, model)
                        print("------------------------")
                        print("Model Name: {}".format(model))
                        print("Local Model Directory: {}".format(model_dir))
                        try:
                            print("Model Files:")
                            pre_level = 1
                            for root, dirs, files in os.walk(model_dir):
                                level = root.replace(model_dir, '').count(os.sep)
                                indent = ' ' * 4 * (level + pre_level)
                                print('{}{}/'.format(indent, os.path.basename(root)))
                                subindent = ' ' * 4 * (level + pre_level + 1)
                                for f in files:
                                    print('{}{}'.format(subindent, f))
                            print("------------------------")
                        except Exception as e:
                            print("------------------------")
                            print("Failed to list the model files. {}".format(e))
                        return [model]
        else:
            return self.list_model_api(model_name, "", api_key)

        return []

    def build_model(self, model_name):
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            return ""

        model_config_file = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE)
        if not os.path.exists(model_config_file):
            print("You model repository is missing file {}, you should add it.".format(
                ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE))
            return ""

        model_bin_file = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_BIN_FILE)
        if not os.path.exists(model_bin_file):
            # User May not upload model bin file, in this case, it means they want to serve a python script
            try:
                with open(model_config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    main_entry_file = config.get("entry_point", "")
                    if main_entry_file == "":
                        print("[Warning] The entry_point is missing in the model config file.")
            except:
                print("You model repository is missing file {}, you should add it.".format(
                    ClientConstants.MODEL_REQUIRED_MODEL_BIN_FILE))
                return ""

        model_readme_file = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_README_FILE)
        if not os.path.exists(model_readme_file):
            print("[Warning] You model repository is missing file {}, we've created an empty README.md for you.".format(
                ClientConstants.MODEL_REQUIRED_MODEL_README_FILE))
            # create a empty readme file called README.md
            with open(model_readme_file, 'w') as f:
                f.write("")
            pass

        if not os.path.exists(ClientConstants.get_model_package_dir()):
            os.makedirs(ClientConstants.get_model_package_dir(), exist_ok=True)

        model_archive_name = os.path.join(ClientConstants.get_model_package_dir(), model_name)
        model_zip_path = "{}.zip".format(model_archive_name)
        if os.path.exists(model_zip_path):
            os.remove(model_zip_path)
        shutil.make_archive(
            model_archive_name,
            "zip",
            root_dir=ClientConstants.get_model_dir(),
            base_dir=model_name,
        )

        if not os.path.exists(model_zip_path):
            return ""

        return model_zip_path

    def push_model(self, model_name, user_id, user_api_key, model_storage_url=None,
                   model_net_url=None, no_uploading_modelops=False, tag_names=None, model_id=None,
                   model_version="v0"):
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            return "", ""

        is_from_open = None
        if model_storage_url is not None:
            is_from_open = True
            model_zip_path = ""
        else:
            is_from_open = False
            model_config_file = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE)
            if not os.path.exists(model_config_file):
                print("You model repository is missing file {}, you should add it.".format(
                    ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE))
                return "", ""

            model_bin_file = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_BIN_FILE)
            if not os.path.exists(model_bin_file):
                # User May not upload model bin file, in this case, it means they want to serve a python script
                try:
                    with open(model_config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        main_entry_file = config.get("entry_point", "")
                        if main_entry_file == "":
                            print("[Warning] The entry_point is missing in the model config file.")
                except:
                    print("You model repository is missing file {}, you should add it.".format(
                        ClientConstants.MODEL_REQUIRED_MODEL_BIN_FILE))
                    return "", ""

            model_readme_file = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_README_FILE)
            if not os.path.exists(model_readme_file):
                print(
                    "[Warning] You model repository is missing file {}, we've created an empty README.md for you."
                    .format(ClientConstants.MODEL_REQUIRED_MODEL_README_FILE))
                # create a empty readme file called README.md
                with open(model_readme_file, 'w') as f:
                    f.write("")
                pass

            if not os.path.exists(ClientConstants.get_model_package_dir()):
                os.makedirs(ClientConstants.get_model_package_dir(), exist_ok=True)

            model_archive_name = os.path.join(ClientConstants.get_model_package_dir(), model_name)
            model_zip_path = "{}.zip".format(model_archive_name)
            if os.path.exists(model_zip_path):
                os.remove(model_zip_path)
            shutil.make_archive(
                model_archive_name,
                "zip",
                root_dir=ClientConstants.get_model_dir(),
                base_dir=model_name,
            )

            if not os.path.exists(model_zip_path):
                return "", ""

            model_storage_url = self.push_model_to_s3(model_name, model_zip_path, user_id)
            print("Model storage url: {}".format(model_storage_url))

        if not no_uploading_modelops:
            if model_storage_url != "":
                with open(model_config_file, 'r') as f:
                    model_params = yaml.safe_load(f)

                upload_result = self.upload_model_api(model_name, model_params, model_storage_url,
                                                      model_net_url, user_id, user_api_key,
                                                      is_from_open=is_from_open, tag_names=tag_names,
                                                      model_id=model_id, version=model_version)
                if upload_result is not None:
                    return model_storage_url, model_zip_path
                else:
                    return "", model_zip_path

        return model_storage_url, model_zip_path

    def pull_model(self, model_name, user_id, user_api_key):
        model_query_result = self.list_model_api(model_name, user_id, user_api_key)
        if model_query_result is None:
            return ""

        local_model_package = ""
        for model in model_query_result.model_list:
            model_storage_url = model.model_url
            query_model_name = model.model_name
            if query_model_name != model_name:
                continue
            if model_storage_url is None or model_storage_url == "":
                continue
            local_model_package = self.pull_model_from_s3(model_storage_url, model_name)
            if local_model_package == "":
                print("Failed to pull model name {}".format(query_model_name))
                return ""
            else:
                # TODO: Check if there are multi models from return from the backend
                print("Successfully pull model name {}".format(query_model_name))
                break

        return local_model_package

    def prepare_yaml_for_launch(self, model_name) -> str:
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            print("[Error] Model {} doesn't exist. Please create it first.".format(model_name))
            return ""

        if not self.recreate_model(model_name):
            print("[Error] Failed to recreate model {}".format(model_name))
            return ""
        else:
            print("Automatically recreate model {}.".format(model_name))

        config_file_path = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE)
        if not os.path.exists(config_file_path):
            print("[Error] Model {} doesn't have config file. Please create it first.".format(model_name))
            return ""

        # Build a tmp launch yaml file
        with open(config_file_path, 'r') as f:
            usr_config_total_params = yaml.safe_load(f)
            usr_resource_req = usr_config_total_params.get("computing", {})

        if usr_resource_req == {}:
            print(
                "[Error] Model {} doesn't have computing resource requirement. Please add it first.".format(model_name))
            return ""

        launch_params = {
            "workspace": model_name,
            "task_type": "serve",
            "job": "",
            "bootstrap": "",
            "computing": usr_resource_req,
        }

        # Dumps a yaml file in the parent directory
        parent_dir = os.path.dirname(model_dir)
        dst_launch_file_name = ClientConstants.MODEL_AUTO_GEN_LAUNCH_FILE
        dst_launch_file_pth = os.path.join(parent_dir, dst_launch_file_name)
        with open(dst_launch_file_pth, 'w') as f:
            yaml.dump(launch_params, f, sort_keys=False)
        return dst_launch_file_pth

    def local_serve_model(self, model_name):
        # Check local model card existence
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            print(f"[Error] Model cards {model_name} doesn't exist. Please create it first.")
            return False

        # Dynamically Recreate Model Cards
        if not self.recreate_model(model_name):
            print("[Error] Failed to recreate model {}".format(model_name))
            return False

        # Parse the config file
        config_file_path = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE)
        config_params = self.parse_config_yaml(config_file_path)
        bootstrap_exec_str = config_params.get("bootstrap", "")
        program_args = config_params.get("program_args", {})
        new_environment_vars = config_params.get("environment_variables", {})

        # Change the execution path to the model dir
        os.chdir(model_dir)
        if bootstrap_exec_str != "":
            print("Executing bootstrap script ...")
            time.sleep(2)
            import subprocess
            process = subprocess.Popen(bootstrap_exec_str, shell=True)
            process.wait()
            print("Bootstrap script is executed successfully!")
            time.sleep(2)

        # Enter main_entry
        main_entry_file = config_params.get("entry_point", "")
        if main_entry_file == "":
            print("The entry_point is missing in the model config file.")
            return False
        main_entry_file = os.path.join(model_dir, main_entry_file)
        if not os.path.exists(main_entry_file):
            print("The entry_point {} doesn't exist.".format(main_entry_file))
            return False

        import subprocess
        all_env_vars = os.environ.copy()
        for k, v in new_environment_vars.items():
            all_env_vars[k] = str(v)

        print(f"Entering the main entry file {main_entry_file} ...")

        extra_args = []
        for k, v in program_args.items():
            extra_args.append(f"--{k}={v}")

        process = subprocess.Popen(
            ["python3", main_entry_file] + extra_args,
            env=all_env_vars,
            )

        print("Local deployment is started. Use Ctrl+C to stop it.")
        try:
            process.wait()
        except KeyboardInterrupt:
            print("Local deployment is stopped.")
            process.kill()
        return True

    def deploy_model(
            self, model_name, device_type, devices, user_id, user_api_key,params, use_local_deployment=None,
            in_model_version=None, in_model_id=None, endpoint_name=None, endpoint_id=None, run_id=None
    ):
        if use_local_deployment is None:
            use_local_deployment = False
        if not use_local_deployment:
            print("Start to query model from Nexus platform ...")
            model_query_result = self.list_model_api(model_name, user_id, user_api_key)
            if model_query_result is None or model_query_result.model_list is None \
                    or len(model_query_result.model_list) == 0:
                print(f"[Error] Failed to query model {model_name} from Nexus platform")
                return False
            for model in model_query_result.model_list:
                model_id = in_model_id if in_model_id is not None and in_model_id != "" else model.id
                model_version = in_model_version if in_model_version is not None and in_model_version != "" \
                    else model.model_version
                print(f"Found {model_name} with model_id: {model_id} and model_version: {model_version}."
                      f"Start to deploy model ...")
                deployment_result = self.deploy_model_api(
                    model_id, model_name, model_version, device_type, devices, user_id, user_api_key,
                    endpoint_name=endpoint_name, endpoint_id=endpoint_id, run_id=run_id)
                if deployment_result is not None:
                    return True
        else:
            print("[MQTT] Start to deploy model locally ...")
            model_id = uuid.uuid4()
            end_point_id = uuid.uuid4()
            end_point_token = "FedMLEndPointToken@{}".format(str(uuid.uuid4()))
            self.send_start_deployment_msg(user_id, user_api_key, end_point_id, end_point_token,
                                           devices, model_name, model_id, params)

        return False

    def query_model(self, model_name):
        return get_model_info(model_name, ClientConstants.INFERENCE_ENGINE_TYPE_ONNX,
                              ClientConstants.INFERENCE_HTTP_PORT)

    def list_model_api(self, model_name, user_id, user_api_key):
        model_list_result = None
        model_ops_url = ClientConstants.get_model_ops_list_url(self.config_version)
        model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        model_list_json = {
            "model_name": model_name,
            "page_num": 1,
            "page_size": 100,
            "user_id": str(user_id),
            "user_api_key": user_api_key
        }
        args = {"config_version": self.config_version}
        _, cert_path = ModelOpsConfigs.get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    model_ops_url, verify=True, headers=model_api_headers, json=model_list_json
                )
            except requests.exceptions.SSLError as err:
                ModelOpsConfigs.install_root_ca_file()
                response = requests.post(
                    model_ops_url, verify=True, headers=model_api_headers, json=model_list_json
                )
        else:
            response = requests.post(model_ops_url, headers=model_api_headers, json=model_list_json)
        if response.status_code != 200:
            pass
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Error: {}.".format(resp_data["message"]))
                return None
            model_list_result = FedMLModelList(resp_data["data"])

        return model_list_result

    def upload_model_api(self, model_name, model_params, model_storage_url, model_net_url,
                         user_id, user_api_key, is_from_open=True,
                         tag_names=None, model_id=None, version="v0"):
        model_upload_result = None
        model_ops_url = ClientConstants.get_model_ops_upload_url(self.config_version)
        model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        tag_list = list()
        if tag_names is not None:
            for name in tag_names:
                tag_list.append({"tagName": name})
        if model_id is not None:
            model_id = int(model_id)
        model_upload_json = {
            "description": model_name,
            "githubLink": "",
            "modelName": model_name,
            "modelUrl": model_storage_url,
            "owner": user_id,
            "parameters": model_params,
            "updateBy": user_id,
            "userId": str(user_id),
            "apiKey": user_api_key,
            "isFromOpen": int(is_from_open),
            "modelNetUrl": model_net_url,
            "tagList": [] if tag_names is None else tag_list,
            "id": model_id,
            "modelVersion": version,
        }
        args = {"config_version": self.config_version}
        _, cert_path = ModelOpsConfigs.get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    model_ops_url, verify=True, headers=model_api_headers, json=model_upload_json
                )
            except requests.exceptions.SSLError as err:
                ModelOpsConfigs.install_root_ca_file()
                response = requests.post(
                    model_ops_url, verify=True, headers=model_api_headers, json=model_upload_json
                )
        else:
            response = requests.post(model_ops_url, headers=model_api_headers, json=model_upload_json)
        if response.status_code != 200:
            pass
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Error: {}.".format(resp_data["message"]))
                return None
            model_upload_result = resp_data

        return model_upload_result

    def update_model_api(self, model_name, model_params, model_storage_url, model_net_url,
                         user_id, user_api_key, is_from_open=True,
                         tag_names=None, model_id=None, version="v0"):
        model_update_result = None
        model_ops_url = ClientConstants.get_model_ops_update_url(self.config_version)
        model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        tag_list = list()
        if tag_names is not None:
            for name in tag_names:
                tag_list.append({"tagName": name})
        if model_id is not None:
            model_id = int(model_id)
        model_upload_json = {
            "description": model_name,
            "githubLink": "",
            "modelName": model_name,
            "modelUrl": model_storage_url,
            "owner": user_id,
            "parameters": model_params,
            "updateBy": user_id,
            "userId": str(user_id),
            "apiKey": user_api_key,
            "isFromOpen": int(is_from_open),
            "modelNetUrl": model_net_url,
            "tagList": [] if tag_names is None else tag_list,
            "id": model_id,
            "modelVersion": version,
        }
        args = {"config_version": self.config_version}
        _, cert_path = ModelOpsConfigs.get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    model_ops_url, verify=True, headers=model_api_headers, json=model_upload_json
                )
            except requests.exceptions.SSLError as err:
                ModelOpsConfigs.install_root_ca_file()
                response = requests.post(
                    model_ops_url, verify=True, headers=model_api_headers, json=model_upload_json
                )
        else:
            response = requests.post(model_ops_url, headers=model_api_headers, json=model_upload_json)
        if response.status_code != 200:
            pass
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Error: {}.".format(resp_data["message"]))
                return None
            model_update_result = resp_data

        return model_update_result

    def push_model_to_s3(self, model_name, model_zip_path, user_id, show_progress=True, progress_desc=None):
        args = {"config_version": self.config_version}
        _, s3_config = ModelOpsConfigs.get_instance(args).fetch_configs(self.config_version)
        s3_storage = S3Storage(s3_config)
        model_dst_key = "{}@{}@{}".format(user_id, model_name, str(uuid.uuid4()))
        model_storage_url = s3_storage.upload_file_with_progress(model_zip_path, model_dst_key,
                                                                 show_progress=show_progress,
                                                                 out_progress_to_err=True,
                                                                 progress_desc=progress_desc)
        return model_storage_url

    def pull_model_from_s3(self, model_storage_url, model_name):
        args = {"config_version": self.config_version}
        _, s3_config = ModelOpsConfigs.get_instance(args).fetch_configs(self.config_version)
        s3_storage = S3Storage(s3_config)
        local_model_package = os.path.join(ClientConstants.get_model_package_dir(), model_name)
        local_model_package = "{}.zip".format(local_model_package)
        print("Pulling......")
        ClientConstants.retrieve_and_unzip_package(model_storage_url,
                                                   model_name,
                                                   local_model_package,
                                                   ClientConstants.get_model_dir())
        if os.path.exists(local_model_package):
            return local_model_package

        return ""

    def deploy_model_api(self, model_id, model_name, model_version, device_type, devices,
                         user_id, user_api_key, endpoint_name=None, endpoint_id=None, run_id=None):
        model_deployment_result = None
        model_ops_url = ClientConstants.get_model_ops_deployment_url(self.config_version)
        model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        if type(devices) is list:
            devices = "[" + ",".join([str(device) for device in devices]) + "]"
        model_deployment_json = {
            "edgeId": devices,
            "endpointName":
                endpoint_name if endpoint_name is not None and endpoint_name != "" else f"EndPoint-{str(uuid.uuid4())}",
            "modelId": model_id,
            "modelVersion": model_version,
            "resourceType": device_type,
            "userId": str(user_id),
            "apiKey": user_api_key,
        }
        if endpoint_id is not None and endpoint_id != "":
            print(f"Updating endpoint {endpoint_id}...")
            model_deployment_json["id"] = int(endpoint_id)
        if run_id is not None:
            model_deployment_json["run_id"] = run_id
        args = {"config_version": self.config_version}
        _, cert_path = ModelOpsConfigs.get_instance(args).get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    model_ops_url, verify=True, headers=model_api_headers, json=model_deployment_json
                )
            except requests.exceptions.SSLError as err:
                ModelOpsConfigs.install_root_ca_file()
                response = requests.post(
                    model_ops_url, verify=True, headers=model_api_headers, json=model_deployment_json
                )
        else:
            response = requests.post(model_ops_url, headers=model_api_headers, json=model_deployment_json)
        if response.status_code != 200:
            print(f"Api error, response data {response.json()}")
            pass
        else:
            print(f"Api Success, response data {response.json()}")
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Error: {}.".format(resp_data["message"]))
                return None
            elif resp_data["code"] == "ENDPOINT_EXIST":
                print("Error: {}.".format(resp_data["message"]))
                return None
            model_deployment_result = resp_data

        return model_deployment_result

    def apply_endpoint_api(self, user_api_key, endpoint_name, endpoint_id=None,
                           model_id=None, model_name=None, model_version=None, run_id=None):
        endpoint_apply_result = None
        model_ops_url = ClientConstants.get_model_ops_apply_endpoint_url(self.config_version)
        endpoint_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        endpoint_apply_json = {
            "apiKey": user_api_key,
            "endpointName": endpoint_name,
            "resourceType": "md.fedml_cloud_device"
        }
        if endpoint_id is not None:
            endpoint_apply_json["id"] = endpoint_id
        if model_id is not None:
            endpoint_apply_json["modelId"] = model_id
        if model_name is not None:
            endpoint_apply_json["modelName"] = model_name
        if model_version is not None:
            endpoint_apply_json["modelVersion"] = model_version
        if run_id is not None:
            endpoint_apply_json["run_id"] = run_id

        args = {"config_version": self.config_version}
        _, cert_path = ModelOpsConfigs.get_instance(args).get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    model_ops_url, verify=True, headers=endpoint_api_headers, json=endpoint_apply_json
                )
            except requests.exceptions.SSLError as err:
                ModelOpsConfigs.install_root_ca_file()
                response = requests.post(
                    model_ops_url, verify=True, headers=endpoint_api_headers, json=endpoint_apply_json
                )
        else:
            response = requests.post(model_ops_url, headers=endpoint_api_headers, json=endpoint_apply_json)
        if response.status_code != 200:
            print(f"Apply endpoint with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Error: {}.".format(resp_data["message"]))
                return None
            endpoint_apply_result = resp_data["data"]
            if endpoint_apply_result is None or endpoint_apply_result == "":
                print(f"Apply endpoint with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return None

        return endpoint_apply_result

    def delete_endpoint_api(self, user_api_key, endpoint_id):
        endpoint_request_result = None
        model_ops_url = ClientConstants.get_model_ops_delete_endpoint_url(self.config_version)
        endpoint_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        endpoint_request_json = {
            "apiKey": user_api_key,
            "endpointId": endpoint_id
        }

        args = {"config_version": self.config_version}
        _, cert_path = ModelOpsConfigs.get_instance(args).get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.post(
                    model_ops_url, verify=True, headers=endpoint_api_headers, json=endpoint_request_json
                )
            except requests.exceptions.SSLError as err:
                ModelOpsConfigs.install_root_ca_file()
                response = requests.post(
                    model_ops_url, verify=True, headers=endpoint_api_headers, json=endpoint_request_json
                )
        else:
            response = requests.post(model_ops_url, headers=endpoint_api_headers, json=endpoint_request_json)
        if response.status_code != 200:
            print(f"Delete endpoint with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Error: {}.".format(resp_data["message"]))
                return None
            endpoint_request_result = resp_data["data"]
            if endpoint_request_result is None or endpoint_request_result == "":
                print(f"Delete endpoint with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return None

        return endpoint_request_result

    def endpoint_inference_api(self, user_api_key, endpoint_id: str, req: str) -> str:
        model_ops_url = ClientConstants.get_model_ops_endpoint_inference_url(endpoint_id)
        endpoint_api_headers = {'Content-Type': 'application/json', 'Connection': 'close',
                                'Authorization': 'Bearer {}'.format(user_api_key)}
        try:
            req_json = json.loads(req)
        except Exception as e:
            print("[Error] Cannot jsonify your req body: {}.".format(e))
            return ""

        try:
            args = {"config_version": self.config_version}
            _, cert_path = ModelOpsConfigs.get_instance(args).get_request_params()

            if cert_path is not None:
                try:
                    requests.session().verify = False
                    response = requests.post(
                        model_ops_url, verify=True, headers=endpoint_api_headers, json=req_json
                    )
                except requests.exceptions.SSLError as err:
                    ModelOpsConfigs.install_root_ca_file()
                    response = requests.post(
                        model_ops_url, verify=True, headers=endpoint_api_headers, json=req_json
                    )
            else:
                response = requests.post(model_ops_url, headers=endpoint_api_headers, json=req_json)
        except Exception as e:
            print("[Error] After post, got: {}.".format(e))
            return ""

        if response.status_code != 200:
            print(f"Endpoint inference with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
            return ""
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Got error msg from mlops: {}.".format(resp_data["message"]))
                return ""
            endpoint_inference_result = resp_data["data"]
            if endpoint_inference_result is None or endpoint_inference_result == "":
                print(f"Endpoint inference with response.status_code = {response.status_code}, "
                      f"response.content: {response.content}")
                return ""
            try:
                endpoint_inference_result = json.dumps(endpoint_inference_result)
            except Exception as e:
                print("[Error] Cannot jsonify the endpoint inference result from mlops: {}.".format(e))
                return ""

        return endpoint_inference_result

    def query_endpoint_detail_api(self, endpoint_name=None, user_api_key=None, endpoint_id=None):
        endpoint_detail_result = None
        if endpoint_id is None:
            model_ops_url = ClientConstants.get_model_ops_endpoint_detail_by_name_url(endpoint_name, config_version=self.config_version)
        else:
            model_ops_url = ClientConstants.get_model_ops_endpoint_detail_url(endpoint_id, config_version=self.config_version)
        endpoint_api_headers = {'Content-Type': 'application/json', 'Connection': 'close',
                                "Authorization": f"Bearer {user_api_key}"}
        args = {"config_version": self.config_version}
        _, cert_path = ModelOpsConfigs.get_request_params()
        if cert_path is not None:
            try:
                requests.session().verify = cert_path
                response = requests.get(
                    model_ops_url, verify=True, headers=endpoint_api_headers,
                )
            except requests.exceptions.SSLError as err:
                ModelOpsConfigs.install_root_ca_file()
                response = requests.get(
                    model_ops_url, verify=True, headers=endpoint_api_headers,
                )
        else:
            response = requests.get(model_ops_url, headers=endpoint_api_headers)
        if response.status_code != 200:
            pass
        else:
            resp_data = response.json()
            if resp_data["code"] != "SUCCESS":
                print("Error: {}.".format(resp_data["message"]))
                return None
            endpoint_detail_result = FedMLEndpointDetail(resp_data["data"])

        return endpoint_detail_result

    def delete_endpoint(self, user_api_key: str, endpoint_id: str) -> bool:
        delete_mlops_url = ClientConstants.get_model_ops_delete_url()
        endpoint_delete_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        endpoint_delete_json = {
            "apiKey": user_api_key,
            "endpointId": int(endpoint_id)
        }

        try:
            args = {"config_version": self.config_version}
            _, cert_path = ModelOpsConfigs.get_instance(args).get_request_params()

            if cert_path is not None:
                try:
                    requests.session().verify = False
                    response = requests.post(
                        delete_mlops_url, verify=True, headers=endpoint_delete_headers, json=endpoint_delete_json
                    )
                except requests.exceptions.SSLError as err:
                    ModelOpsConfigs.install_root_ca_file()
                    response = requests.post(
                        delete_mlops_url, verify=True, headers=endpoint_delete_headers, json=endpoint_delete_json
                    )
            else:
                response = requests.post(delete_mlops_url, headers=endpoint_delete_headers, json=endpoint_delete_json)
        except Exception as e:
            print("[Error] After post req, got: {}.".format(e))
            return False

        if response.status_code != 200:
            print(f"Delete endpoint with response.status_code = {response.status_code}, "
                  f"response.content: {response.content}")
            return False
        return True

    def send_start_deployment_msg(self, user_id, user_api_key, end_point_id, end_point_token,
                                  devices, model_name, model_id, params):
        ServerConstants.get_local_ip()
        device_id_list = json.loads(devices)
        device_objs = list()
        cur_index = 0
        for device_id in device_id_list:
            device_id_str = ""
            uuid_prefix = str(uuid.uuid4())
            if cur_index == 0:
                device_id_str = "{}@MacOS.MDA.OnPremise.Master.Device".format(uuid_prefix)
            else:
                device_id_str = "{}@MacOS.MDA.OnPremise.Device".format(uuid_prefix)
            cur_index += 1

            device_objs.append({"device_id": device_id_str,
                                "os_type": "MacOS", "id": device_id, "ip": "0.0.0.0",
                                "memory": "64G", "cpu": "2.7", "gpu": "AppleM1", "extra_infos": {}})

        model_storage_url, _ = self.push_model(model_name, user_id, user_api_key, no_uploading_modelops=True)

        master_device_id = device_id_list[0]
        topic_start_deployment = "model_ops/model_device/start_deployment/{}".format(str(master_device_id))
        start_deployment_payload = {"timestamp": int(time.time()), "end_point_id": str(end_point_id),
                                    "token": str(end_point_token), "state": "STARTING", "user_id": user_id,
                                    "user_name": user_id,
                                    "device_ids": device_id_list,
                                    "device_objs": device_objs,
                                    "model_config": {"model_name": model_name, "model_id": str(model_id),
                                                     "model_version": "v0-Fri Jan 06 06:36:44 GMT 2023",
                                                     "model_storage_url": model_storage_url,
                                                     "instance_scale_min": 1, "instance_scale_max": 3,
                                                     "inference_engine": ClientConstants.INFERENCE_ENGINE_TYPE_ONNX},
                                    "parameters": params}

        self.local_deployment_end_point_id = end_point_id
        args = {"config_version": "release"}
        mqtt_config, _ = ModelOpsConfigs.get_instance(args).fetch_configs(self.config_version)
        self.local_deployment_mqtt_mgr = MqttManager(
            mqtt_config["BROKER_HOST"],
            mqtt_config["BROKER_PORT"],
            mqtt_config["MQTT_USER"],
            mqtt_config["MQTT_PWD"],
            mqtt_config["MQTT_KEEPALIVE"],
            "FedML_LocalModelDeployment_" + str(uuid.uuid4())
        )
        self.local_deployment_mqtt_mgr.add_connected_listener(self.on_mqtt_connected)
        self.local_deployment_mqtt_mgr.add_disconnected_listener(self.on_mqtt_disconnected)
        self.local_deployment_mqtt_mgr.connect()
        self.local_deployment_mqtt_mgr.loop_start()
        self.local_deployment_mqtt_mgr.send_message_json(topic_start_deployment, json.dumps(start_deployment_payload))

        while not self.should_end_local_deployment:
            time.sleep(1)

        time.sleep(300)
        self.local_deployment_mqtt_mgr.disconnect()
        self.local_deployment_mqtt_mgr.loop_stop()

    def on_mqtt_connected(self, mqtt_client_object):
        deployment_results_topic = "model_ops/model_device/return_deployment_result/{}".format(
            self.local_deployment_end_point_id)
        deployment_status_topic = "model_ops/model_device/return_deployment_status/{}".format(
            self.local_deployment_end_point_id)
        deployment_stages_topic = "model_ops/model_device/return_deployment_stages/{}".format(
            self.local_deployment_end_point_id)
        deployment_monitoring_topic = "model_ops/model_device/return_inference_monitoring/{}".format(
            self.local_deployment_end_point_id)

        self.local_deployment_mqtt_mgr.add_message_listener(deployment_results_topic,
                                                            self.callback_deployment_results_msg)
        self.local_deployment_mqtt_mgr.add_message_listener(deployment_status_topic,
                                                            self.callback_deployment_status_msg)
        self.local_deployment_mqtt_mgr.add_message_listener(deployment_stages_topic,
                                                            self.callback_deployment_stages_msg)
        self.local_deployment_mqtt_mgr.add_message_listener(deployment_monitoring_topic,
                                                            self.callback_deployment_monitoring_msg)

        mqtt_client_object.subscribe(deployment_results_topic, qos=2)
        mqtt_client_object.subscribe(deployment_status_topic, qos=2)
        mqtt_client_object.subscribe(deployment_stages_topic, qos=2)
        mqtt_client_object.subscribe(deployment_monitoring_topic, qos=2)

    def on_mqtt_disconnected(self, mqtt_client_object):
        pass

    def callback_deployment_results_msg(self, topic, payload):
        print("--------------------------------------------------")
        print("--------Received deployment result message--------")
        print("Message topic: {}".format(topic))
        print("Message payload: {}".format(payload))

    def callback_deployment_status_msg(self, topic, payload):
        print("--------------------------------------------------")
        print("--------Received deployment status message--------")
        print("Message topic: {}".format(topic))
        print("Message payload: {}".format(payload))

        payload_json = json.loads(payload)
        status = payload_json["model_status"]
        if status == ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED or \
                status == ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_FAILED:
            self.should_end_local_deployment = True

    def callback_deployment_stages_msg(self, topic, payload):
        print("--------------------------------------------------")
        print("--------Received deployment stages message--------")
        print("Message topic: {}".format(topic))
        print("Message payload: {}".format(payload))

    def callback_deployment_monitoring_msg(self, topic, payload):
        print("--------------------------------------------------")
        print("--------Received deployment monitoring message--------")
        print("Message topic: {}".format(topic))
        print("Message payload: {}".format(payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cf", "-c", help="config file")
    parser.add_argument("--role", "-r", type=str, default="client", help="role")
    in_args = parser.parse_args()
    # fedml.set_env_version("dev")
    # FedMLModelCards.get_instance().deploy_model_api(
    #     "375", "mnist", "v0-Thu Nov 16 09:41:42 GMT 2023",
    #     "md.on_premise_device", [1,2], "XXAA",
    #     "XXAA", endpoint_name="EndPoint-a8a8c0b0-cb3f-428a-a4f3-6e9e2437a096"
    # )
