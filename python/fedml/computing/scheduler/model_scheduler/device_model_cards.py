import argparse
import json
import os
import shutil
import time
import uuid
import yaml

import requests
from ....core.distributed.communication.mqtt.mqtt_manager import MqttManager

from ....core.distributed.communication.s3.remote_storage import S3Storage

from .device_client_constants import ClientConstants
from ....core.common.singleton import Singleton
from .modelops_configs import ModelOpsConfigs
from .device_model_deployment import get_model_info
from .device_server_constants import ServerConstants
from .device_model_object import FedMLModelList
from .device_client_constants import ClientConstants


class FedMLModelCards(Singleton):

    def __init__(self):
        self.current_model_name = None
        self.local_deployment_end_point_id = None
        self.should_end_local_deployment = False
        self.local_deployment_mqtt_mgr = None

    @staticmethod
    def get_instance():
        return FedMLModelCards()

    def serve_model(self, model_name):
        src_folder = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(src_folder):
            print("Model {} doesn't exist. Please Create it First".format(model_name))
            return False
        config_file_path = os.path.join(src_folder, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE)
        parms_dict = self.parse_config_yaml(config_file_path)

        user_id = parms_dict.get("FEDML_USER_ID", os.environ.get("FEDML_USER_ID", None))
        user_api_key = parms_dict.get("FEDML_API_KEY", os.environ.get("FEDML_API_KEY", None))
        device_type = parms_dict.get("device_type", "md.on_premise_device")
        master_device_id = parms_dict.get("FEDML_MODEL_SERVE_MASTER_DEVICE_IDS",
                                          os.environ.get("FEDML_MODEL_SERVE_MASTER_DEVICE_IDS", None))
        worker_device_ids = parms_dict.get("FEDML_MODEL_SERVE_WORKER_DEVICE_IDS",
                                           os.environ.get("FEDML_MODEL_SERVE_WORKER_DEVICE_IDS", None))
        additional_parms_dict = parms_dict.get("default_parms_dict", {})
        use_local_deployment = parms_dict.get("default_use_local", False)
        local_server = parms_dict.get("local_server", "127.0.0.1")
        mlops_version = parms_dict.get("mlops_version", "release")

        if master_device_id is None or worker_device_ids is None:
            print("Please specify the master device id and worker device ids in the config file.")
            return False
        if not (type(worker_device_ids) in [str, list, int] and type(master_device_id) in [str, int, list]):
            print('''The format of worker_device_ids is wrong, 
                  please use formate like 1,2,3 or 1,
                  E.g. export FEDML_MODEL_SERVE_WORKER_DEVICE_IDS=1,2,3
                  ''')
            return False

        if type(worker_device_ids) is not list:
            if type(worker_device_ids) is int:
                worker_device_ids = [worker_device_ids]
            else:
                worker_device_ids = worker_device_ids.split(",")
        if type(master_device_id) is not list:
            master_device_id = [str(master_device_id)]
        devices = master_device_id + worker_device_ids

        self.set_config_version(mlops_version)
        self.build_model(model_name)
        self.push_model(model_name, user_id, user_api_key)
        res = self.deploy_model(model_name, device_type, devices, user_id, user_api_key,
                                additional_parms_dict, use_local_deployment, local_server)
        if not res:
            print("Failed to deploy model")
            return False

    def parse_config_yaml(self, yaml_file):
        with open(yaml_file, 'r') as f:
            launch_params = yaml.safe_load(f)
        return launch_params

    def copy_config_yaml_to_src_folder(self, src_folder, yaml_file):
        shutil.copy(yaml_file, os.path.join(src_folder, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE))
        # Change the workspace to "./"
        with open(os.path.join(src_folder, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE), 'r') as f:
            launch_params = yaml.safe_load(f)
            launch_params["workspace"] = "./"
        with open(os.path.join(src_folder, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE), 'w') as f:
            yaml.dump(launch_params, f, sort_keys=False)
        return True

    def set_config_version(self, config_version):
        if config_version is not None:
            self.config_version = config_version

    def create_model_use_config(self, model_name, config_file) -> bool:
        if not os.path.exists(config_file):
            print("The config file {} doesn't exist.".format(config_file))
            return False

        model_config = self.parse_config_yaml(config_file)
        # Copy workspace dir to local model cards dir
        if "workspace" not in model_config:
            print("Please specify the workspace in the config file.")
            return False

        if self.add_model_files(model_name, model_config["workspace"]):
            model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
            if self.copy_config_yaml_to_src_folder(model_dir, config_file):
                return True
            else:
                print(f"Failed to add your config file {config_file} to the model {model_name}.")
                return False
        else:
            print(f"Failed to add your workspace {model_config['workspace']} to the model {model_name}.")
            return False

    def create_model(self, model_name):
        self.current_model_name = model_name
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        return True

    def delete_model(self, model_name):
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
        else:
            return False
        return True

    def add_model_files(self, model_name, file_path):
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            self.create_model(model_name)

        if not os.path.exists(model_dir):
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

    def list_models(self, model_name, user_id=None, user_api_key=None, local_server=None):
        if user_id is None:
            model_home_dir = ClientConstants.get_model_dir()
            if not os.path.exists(model_home_dir):
                return []

            models = os.listdir(model_home_dir)
            if model_name == "*":
                return models
            else:
                for model in models:
                    if model == model_name:
                        return [model]
        else:
            return self.list_model_api(model_name, user_id, user_api_key, local_server)

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
                        print("The entry_point is missing in the model config file.")
                        return ""
            except:
                print("You model repository is missing file {}, you should add it.".format(
                    ClientConstants.MODEL_REQUIRED_MODEL_BIN_FILE))
                return ""

        model_readme_file = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_README_FILE)
        if not os.path.exists(model_readme_file):
            print("You model repository is missing file {}, we've created an empty README.md for you.".format(
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
                   model_net_url=None, no_uploading_modelops=False, local_server=None):
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
                            print("The entry_point is missing in the model config file.")
                            return "", ""
                except:
                    print("You model repository is missing file {}, you should add it.".format(
                        ClientConstants.MODEL_REQUIRED_MODEL_BIN_FILE))
                    return "", ""

            model_readme_file = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_README_FILE)
            if not os.path.exists(model_readme_file):
                print("You model repository is missing file {}, we've created an empty README.md for you.".format(
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
                return "", ""

            model_storage_url = self.push_model_to_s3(model_name, model_zip_path, user_id)
            print("Model storage url: {}".format(model_storage_url))

        if not no_uploading_modelops:
            if model_storage_url != "":
                with open(model_config_file, 'r') as f:
                    model_params = yaml.safe_load(f)

                upload_result = self.upload_model_api(model_name, model_params, model_storage_url,
                                                      model_net_url, user_id, user_api_key,
                                                      is_from_open=is_from_open, local_server=local_server)
                if upload_result is not None:
                    return model_storage_url, model_zip_path
                else:
                    return "", model_zip_path

        return model_storage_url, model_zip_path

    def pull_model(self, model_name, user_id, user_api_key, local_server=None):
        model_query_result = self.list_model_api(model_name, user_id, user_api_key, local_server=local_server)
        if model_query_result is None:
            return False

        result = True
        for model in model_query_result.model_list:
            model_storage_url = model.model_url
            query_model_name = model.model_name
            if query_model_name != model_name:
                continue
            if model_storage_url is None or model_storage_url == "":
                continue
            local_model_package = self.pull_model_from_s3(model_storage_url, model_name)
            if local_model_package == "":
                result = False
                print("Failed to pull model name {}".format(query_model_name))

        return result

    def find_yaml_for_launch(self, model_name) -> str:
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            print("Model {} doesn't exist. Please Create it First".format(model_name))
            return ""

        config_file_path = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE)
        if not os.path.exists(config_file_path):
            print("Model {} doesn't have config file. Please Create it First".format(model_name))
            return ""

        # add some default parameters for fedml®launch to read
        addtional_parms = {
            "task_type": "serve"
        }
        with open(config_file_path, 'r') as f:
            config_parms = yaml.safe_load(f)
            config_parms.update(addtional_parms)
        # Keep the same file name
        with open(config_file_path, 'w') as f:
            yaml.dump(config_parms, f, sort_keys=False)

        print("The config file {} is used for launching.".format(config_file_path))
        return config_file_path

    def local_serve_model(self, model_name):
        # Check local model card existance
        model_dir = os.path.join(ClientConstants.get_model_dir(), model_name)
        if not os.path.exists(model_dir):
            print("Model {} doesn't exist. Please Create it First".format(model_name))
            return False

        # Execute bootstrap script
        config_file_path = os.path.join(model_dir, ClientConstants.MODEL_REQUIRED_MODEL_CONFIG_FILE)
        config_parms = self.parse_config_yaml(config_file_path)
        bootstrap_path = config_parms.get("bootstrap_path", None)
        if bootstrap_path is not None:
            dir_name, file_name = os.path.split(bootstrap_path)
            if ClientConstants.run_bootstrap(dir_name, file_name):
                print("Bootstrap script {} is executed successfully.".format(bootstrap_path))
            else:
                print("Failed to execute bootstrap script {}".format(bootstrap_path))
                return False

        # Enter main_entry
        main_entry_file = config_parms.get("entry_point", "")
        if main_entry_file == "":
            print("The entry_point is missing in the model config file.")
            return False
        main_entry_file = os.path.join(model_dir, main_entry_file)
        if not os.path.exists(main_entry_file):
            print("The entry_point {} doesn't exist.".format(main_entry_file))
            return False
        import subprocess
        # Change the execution path to the model dir
        os.chdir(model_dir)
        process = subprocess.Popen(["python3", main_entry_file])
        print("Local deployment is started. Use Ctrl+C to stop it.")
        try:
            process.wait()
        except KeyboardInterrupt:
            print("Local deployment is stopped.")
            process.kill()
        return True

    def deploy_model(self, model_name, device_type, devices, user_id, user_api_key,
                     params, use_local_deployment=None, local_server=None,
                     in_model_version=None, endpoint_name=None):
        if use_local_deployment is None:
            use_local_deployment = False
        if not use_local_deployment:
            model_query_result = self.list_model_api(model_name, user_id, user_api_key, local_server)
            if model_query_result is None:
                return False
            for model in model_query_result.model_list:
                model_id = model.id
                model_version = in_model_version if in_model_version is not None and in_model_version != "" \
                    else model.model_version
                deployment_result = self.deploy_model_api(model_id, model_name, model_version, device_type,
                                                          devices, user_id, user_api_key, local_server,
                                                          endpoint_name=endpoint_name)
                if deployment_result is not None:
                    return True
        else:
            model_id = uuid.uuid4()
            end_point_id = uuid.uuid4()
            end_point_token = "FedMLEndPointToken@{}".format(str(uuid.uuid4()))
            self.send_start_deployment_msg(user_id, user_api_key, end_point_id, end_point_token,
                                           devices, model_name, model_id, params)

        return False

    def query_model(self, model_name):
        return get_model_info(model_name, ClientConstants.INFERENCE_HTTP_PORT)

    def list_model_api(self, model_name, user_id, user_api_key, local_server):
        model_list_result = None
        model_ops_url = ClientConstants.get_model_ops_list_url(self.config_version, local_server)
        model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
        model_list_json = {
            "model_name": model_name,
            "page_num": 1,
            "page_size": 100,
            "user_id": str(user_id),
            "user_api_key": user_api_key
        }
        args = {"config_version": self.config_version}
        _, cert_path = ModelOpsConfigs.get_instance(args).get_request_params(self.config_version)
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
                         user_id, user_api_key, is_from_open=True, local_server=None):
        model_upload_result = None
        model_ops_url = ClientConstants.get_model_ops_upload_url(self.config_version, local_server)
        model_api_headers = {'Content-Type': 'application/json', 'Connection': 'close'}
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
            "modelNetUrl": model_net_url
        }
        args = {"config_version": self.config_version}
        _, cert_path = ModelOpsConfigs.get_instance(args).get_request_params(self.config_version)
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

    def push_model_to_s3(self, model_name, model_zip_path, user_id, progress_desc=None):
        args = {"config_version": self.config_version}
        _, s3_config = ModelOpsConfigs.get_instance(args).fetch_configs(self.config_version)
        s3_storage = S3Storage(s3_config)
        model_dst_key = "{}@{}@{}".format(user_id, model_name, str(uuid.uuid4()))
        model_storage_url = s3_storage.upload_file_with_progress(model_zip_path, model_dst_key,
                                                                 out_progress_to_err=False,
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
                         user_id, user_api_key, local_server, endpoint_name=None):
        model_deployment_result = None
        model_ops_url = ClientConstants.get_model_ops_deployment_url(self.config_version, local_server)
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
            "apiKey": user_api_key
        }
        args = {"config_version": self.config_version}
        _, cert_path = ModelOpsConfigs.get_instance(args).get_request_params(self.config_version)
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
            pass
        else:
            resp_data = response.json()
            if resp_data["code"] == "FAILURE":
                print("Error: {}.".format(resp_data["message"]))
                return None
            model_deployment_result = resp_data

        return model_deployment_result

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
